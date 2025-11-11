#!/usr/bin/env python3
"""
Database Migration: Add Furniture Color Storage
Adds color_palette column to furniture_detections table
"""

import duckdb
import argparse
from pathlib import Path


def migrate_add_furniture_colors(db_path: str, dry_run: bool = False):
    """
    Add furniture color storage to database schema

    Adds:
    - color_palette: JSON column in furniture_detections table
      Stores per-furniture color information extracted from SAM2 masks

    Args:
        db_path: Path to DuckDB database
        dry_run: If True, only show what would be done
    """

    print("=" * 70)
    print("🔄 DATABASE MIGRATION: Add Furniture Colors")
    print("=" * 70)

    db_path = Path(db_path)

    if not db_path.exists():
        print(f"\n❌ Database not found: {db_path}")
        return False

    print(f"\n📁 Database: {db_path}")

    # Connect to database
    conn = duckdb.connect(str(db_path))

    # Check current schema
    print("\n📊 Current Schema:")
    columns = conn.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'furniture_detections'
        ORDER BY ordinal_position
    """).fetchall()

    print("   furniture_detections table:")
    for col_name, col_type in columns:
        print(f"      {col_name}: {col_type}")

    # Check if color_palette already exists
    has_color_palette = any(col[0] == 'color_palette' for col in columns)

    if has_color_palette:
        print("\n✅ color_palette column already exists - no migration needed")
        conn.close()
        return True

    # Migration SQL
    migration_sql = """
        ALTER TABLE furniture_detections
        ADD COLUMN color_palette JSON
    """

    if dry_run:
        print("\n🔍 DRY RUN - Would execute:")
        print(migration_sql)
        print("\n✅ Dry run complete - no changes made")
        conn.close()
        return True

    # Execute migration
    print("\n🔧 Executing migration...")

    try:
        conn.execute(migration_sql)
        conn.commit()
        print("✅ Migration successful!")

        # Verify
        print("\n🔍 Verifying migration...")
        columns_after = conn.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'furniture_detections'
            AND column_name = 'color_palette'
        """).fetchall()

        if columns_after:
            print("✅ color_palette column added successfully")
            print(f"   Type: {columns_after[0][1]}")
        else:
            print("⚠️  Verification failed - column not found")

        # Show updated schema
        print("\n📊 Updated Schema:")
        all_columns = conn.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'furniture_detections'
            ORDER BY ordinal_position
        """).fetchall()

        print("   furniture_detections table:")
        for col_name, col_type in all_columns:
            marker = " ✨ NEW" if col_name == 'color_palette' else ""
            print(f"      {col_name}: {col_type}{marker}")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        conn.close()
        return False

    conn.close()

    print("\n" + "=" * 70)
    print("✅ MIGRATION COMPLETE")
    print("=" * 70)

    print("\n📝 Next Steps:")
    print("   1. Run color extraction to populate color_palette:")
    print(f"      python src/models/color_extractor.py --db {db_path} --batch")
    print("\n   2. Train mask-enhanced classifier:")
    print(f"      python src/models/mask_enhanced_style_classifier.py --db {db_path}")

    return True


def rollback_migration(db_path: str):
    """
    Rollback the migration (remove color_palette column)

    Args:
        db_path: Path to DuckDB database
    """

    print("=" * 70)
    print("⏮️  DATABASE ROLLBACK: Remove Furniture Colors")
    print("=" * 70)

    db_path = Path(db_path)

    if not db_path.exists():
        print(f"\n❌ Database not found: {db_path}")
        return False

    print(f"\n📁 Database: {db_path}")

    conn = duckdb.connect(str(db_path))

    # Check if column exists
    columns = conn.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'furniture_detections'
        AND column_name = 'color_palette'
    """).fetchall()

    if not columns:
        print("\n✅ color_palette column doesn't exist - nothing to rollback")
        conn.close()
        return True

    print("\n🔧 Removing color_palette column...")

    try:
        # DuckDB doesn't support DROP COLUMN directly, need to recreate table
        # Get all data
        print("   Backing up data...")
        backup_data = conn.execute("""
            SELECT
                detection_id, image_id, item_type, confidence,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                area_percentage, mask_area, mask_score, has_mask
            FROM furniture_detections
        """).fetchall()

        print(f"   Backed up {len(backup_data)} rows")

        # Drop and recreate table
        print("   Recreating table...")
        conn.execute("DROP TABLE furniture_detections")

        conn.execute("""
            CREATE TABLE furniture_detections (
                detection_id VARCHAR PRIMARY KEY,
                image_id VARCHAR,
                item_type VARCHAR,
                confidence FLOAT,
                bbox_x1 FLOAT,
                bbox_y1 FLOAT,
                bbox_x2 FLOAT,
                bbox_y2 FLOAT,
                area_percentage FLOAT,
                mask_area INTEGER,
                mask_score FLOAT,
                has_mask BOOLEAN
            )
        """)

        # Restore data
        print("   Restoring data...")
        for row in backup_data:
            conn.execute("""
                INSERT INTO furniture_detections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, row)

        conn.commit()

        print(f"✅ Rollback successful - restored {len(backup_data)} rows")

    except Exception as e:
        print(f"\n❌ Rollback failed: {e}")
        conn.close()
        return False

    conn.close()

    print("\n" + "=" * 70)
    print("✅ ROLLBACK COMPLETE")
    print("=" * 70)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Database migration to add furniture color storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview migration (dry run)
  python scripts/migrate_db_add_furniture_colors.py \\
      --db database_r2_full.duckdb \\
      --dry-run

  # Execute migration
  python scripts/migrate_db_add_furniture_colors.py \\
      --db database_r2_full.duckdb

  # Rollback migration
  python scripts/migrate_db_add_furniture_colors.py \\
      --db database_r2_full.duckdb \\
      --rollback
        """
    )

    parser.add_argument('--db', type=str, required=True,
                        help='Path to DuckDB database')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview migration without making changes')
    parser.add_argument('--rollback', action='store_true',
                        help='Rollback migration (remove color_palette column)')

    args = parser.parse_args()

    if args.rollback:
        success = rollback_migration(args.db)
    else:
        success = migrate_add_furniture_colors(args.db, dry_run=args.dry_run)

    exit(0 if success else 1)
