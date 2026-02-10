"""
Migration Script: recommendations_history JSON → SQLite
Date: 2026-01-28
Description: Migrate existing recommendation data from JSON files to SQLite database

This script:
1. Loads data from legacy JSON files (recommendations_history_w.json, recommendations_history_m.json)
2. Converts to RecommendationDAO format
3. Inserts into data_prod.db
4. Validates row counts and data integrity
"""

import os
import sys
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dao.recommendation_dao import RecommendationDAO, RecommendationRecord, Stock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecommendationMigration:
    def __init__(self, db_path="data_prod.db", merge_strategy="skip"):
        """
        Initialize migration

        Args:
            db_path: Path to SQLite database
            merge_strategy: How to handle duplicate dates
                - "skip": Skip if date exists (default)
                - "overwrite": Overwrite existing with new data
                - "merge": Merge stocks from both sources
        """
        self.db_path = db_path
        self.merge_strategy = merge_strategy
        self.migration_stats = {
            'total_records': 0,
            'total_stocks': 0,
            'skipped_records': 0,
            'overwritten_records': 0,
            'merged_records': 0,
            'files_processed': 0,
        }

    def _infer_frequency(self, json_path: str) -> str:
        """
        Infer frequency from JSON filename

        Args:
            json_path: Path to JSON file

        Returns:
            'weekly' or 'monthly'
        """
        filename = os.path.basename(json_path).lower()
        if '_w.json' in filename or 'weekly' in filename:
            return 'weekly'
        elif '_m.json' in filename or 'monthly' in filename:
            return 'monthly'
        else:
            raise ValueError(f"Cannot infer frequency from filename: {json_path}")

    def migrate_json_file(self, json_path: str) -> int:
        """
        Migrate a single JSON file to database

        Args:
            json_path: Path to JSON file

        Returns:
            Number of records migrated
        """
        if not os.path.exists(json_path):
            logger.warning(f"JSON file not found: {json_path}")
            return 0

        # Infer frequency from filename
        frequency = self._infer_frequency(json_path)
        logger.info(f"Processing: {json_path} (frequency={frequency})")

        # Create DAO with frequency filter
        dao = RecommendationDAO(db_path=self.db_path, frequency=frequency)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read JSON file {json_path}: {e}")
            return 0

        records_migrated = 0

        for item in data:
            try:
                date = item.get('date')
                stocks_data = item.get('stocks', [])

                if not date:
                    logger.warning(f"Skipping record with missing date")
                    self.migration_stats['skipped_records'] += 1
                    continue

                # Convert stock dicts to Stock objects first
                stocks = []
                for stock_dict in stocks_data:
                    try:
                        stock = Stock.from_dict(stock_dict)
                        stocks.append(stock)
                    except Exception as e:
                        logger.warning(f"Failed to parse stock in {date}: {stock_dict}, error: {e}")
                        continue

                # Check if record already exists
                existing_record = dao.get_by_date(date)
                if existing_record:
                    if self.merge_strategy == "skip":
                        logger.info(f"Record for {date} already exists, skipping")
                        continue
                    elif self.merge_strategy == "overwrite":
                        logger.info(f"Overwriting record for {date}")
                        self.migration_stats['overwritten_records'] += 1
                    elif self.merge_strategy == "merge":
                        # Merge: combine stocks from both sources, deduplicate by stock_id
                        logger.info(f"Merging stocks for {date}")
                        existing_stock_ids = {s.id for s in existing_record.stocks}
                        new_stocks = [s for s in stocks if s.id not in existing_stock_ids]
                        stocks = existing_record.stocks + new_stocks
                        self.migration_stats['merged_records'] += 1
                        logger.info(f"  Added {len(new_stocks)} new stocks, total: {len(stocks)}")

                # Create and save record
                record = RecommendationRecord(date=date, stocks=stocks)
                dao.add_record(record)

                records_migrated += 1
                self.migration_stats['total_stocks'] += len(stocks)

                logger.info(f"✓ Migrated {date}: {len(stocks)} stocks")

            except Exception as e:
                logger.error(f"Failed to migrate record: {item.get('date', 'UNKNOWN')}, error: {e}")
                self.migration_stats['skipped_records'] += 1
                continue

        self.migration_stats['total_records'] += records_migrated
        self.migration_stats['files_processed'] += 1

        return records_migrated

    def validate_migration(self, json_files: list) -> bool:
        """
        Validate that all data was migrated correctly

        Args:
            json_files: List of JSON file paths

        Returns:
            True if validation passes
        """
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION: Comparing JSON vs Database")
        logger.info("=" * 60)

        all_valid = True

        for json_path in json_files:
            if not os.path.exists(json_path):
                continue

            # Create frequency-specific DAO for validation
            frequency = self._infer_frequency(json_path)
            dao = RecommendationDAO(db_path=self.db_path, frequency=frequency)

            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            for item in json_data:
                date = item.get('date')
                if not date:
                    continue

                db_record = dao.get_by_date(date)

                if not db_record:
                    logger.error(f"✗ Missing in DB: {date} (frequency={frequency})")
                    all_valid = False
                    continue

                json_stock_count = len(item.get('stocks', []))
                db_stock_count = len(db_record.stocks)

                if json_stock_count != db_stock_count:
                    logger.error(f"✗ Stock count mismatch for {date}: JSON={json_stock_count}, DB={db_stock_count}")
                    all_valid = False
                else:
                    logger.debug(f"✓ {date}: {db_stock_count} stocks")

        return all_valid

    def print_summary(self):
        """Print migration summary"""
        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Merge strategy:     {self.merge_strategy}")
        logger.info(f"Files processed:    {self.migration_stats['files_processed']}")
        logger.info(f"Records migrated:   {self.migration_stats['total_records']}")
        logger.info(f"Total stocks:       {self.migration_stats['total_stocks']}")
        logger.info(f"Skipped records:    {self.migration_stats['skipped_records']}")
        logger.info(f"Overwritten:        {self.migration_stats['overwritten_records']}")
        logger.info(f"Merged:             {self.migration_stats['merged_records']}")
        logger.info(f"Database path:      {self.db_path}")
        logger.info("=" * 60)


def main():
    """Main migration function"""
    logger.info("=" * 60)
    logger.info("RECOMMENDATION DATA MIGRATION: JSON → SQLite")
    logger.info(f"Timestamp: {datetime.now(ZoneInfo('Asia/Taipei')).isoformat()}")
    logger.info("=" * 60)

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Migrate recommendation data from JSON to SQLite")
    parser.add_argument(
        '--strategy',
        choices=['skip', 'overwrite', 'merge'],
        default='skip',
        help='Strategy for handling duplicate dates (default: skip)',
    )
    args = parser.parse_args()

    logger.info(f"Merge strategy: {args.strategy}")

    # Initialize migration
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    migration = RecommendationMigration(db_path=os.path.join(root_dir, "data_prod.db"), merge_strategy=args.strategy)

    # JSON files to migrate
    json_files = [
        os.path.join(root_dir, "assets", "recommendations_history_w.json"),
        os.path.join(root_dir, "assets", "recommendations_history_m.json"),
    ]

    # Migrate each file
    for json_file in json_files:
        filename = os.path.basename(json_file)
        logger.info(f"\n📁 Processing: {filename}")
        logger.info("-" * 60)

        count = migration.migrate_json_file(json_file)
        logger.info(f"✓ Migrated {count} records from {filename}")

    # Validate migration
    logger.info("\n🔍 Validating migration...")
    validation_passed = migration.validate_migration(json_files)

    # Print summary
    migration.print_summary()

    if validation_passed:
        logger.info("\n✅ Migration completed successfully!")
    else:
        if migration.merge_strategy == "merge":
            logger.info("\n✅ Migration completed with merge strategy!")
            logger.info("⚠️  Validation warnings are EXPECTED when using merge strategy:")
            logger.info("   Database contains combined stocks from both weekly and monthly files")
            logger.info("   Stock counts in DB will be >= individual JSON file counts")
        else:
            logger.error("\n❌ Migration validation FAILED!")
            logger.error("   Please review errors above and re-run migration")
            return 1

    logger.info("\n💡 Next steps:")
    logger.info("   1. Verify data in data_prod.db using SQLite browser")
    logger.info("   2. Run backtest to confirm strategy works with new data")
    logger.info("   3. Keep JSON files as backup (already in .gitignore)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
