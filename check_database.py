#!/usr/bin/env python3
"""
Quick Database Checker
Verifies your database is ready for Phase 2 training
"""

import os
import sys

print("=" * 70)
print("DATABASE VERIFICATION SCRIPT")
print("=" * 70)

# Check if database exists
db_path = "database_metadata.duckdb"
if not os.path.exists(db_path):
    print(f"\n❌ Database not found at: {db_path}")
    print("\nSearching for database files...")
    import glob
    db_files = glob.glob("**/*.duckdb", recursive=True)
    if db_files:
        print(f"Found database files:")
        for f in db_files:
            print(f"  - {f}")
    else:
        print("No .duckdb files found in project")
    sys.exit(1)

print(f"\n✅ Database found: {db_path}")
print(f"   Size: {os.path.getsize(db_path) / 1024 / 1024:.1f} MB")

# Try to read database
try:
    import duckdb
except ImportError:
    print("\n⚠️  DuckDB not installed. Installing...")
    print("   Run: pip install duckdb")
    sys.exit(1)

try:
    print("\n📊 Connecting to database...")
    conn = duckdb.connect(db_path, read_only=True)

    # Get tables
    tables_result = conn.execute("SHOW TABLES").fetchall()
    tables = [t[0] for t in tables_result]

    print(f"   Tables found: {len(tables)}")

    if not tables:
        print("\n❌ Database is empty (no tables)")
        conn.close()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("TABLE CONTENTS")
    print("=" * 70)

    total_images = 0
    total_detections = 0

    for table_name in tables:
        print(f"\n📋 Table: {table_name}")
        print("-" * 70)

        # Get row count
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"   Rows: {count:,}")

        if table_name.lower() == 'images':
            total_images = count
        elif 'detection' in table_name.lower():
            total_detections = count

        # Get columns
        schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
        print(f"   Columns: {len(schema)}")
        for col in schema[:10]:  # Show first 10 columns
            print(f"      - {col[0]:25s} {col[1]}")
        if len(schema) > 10:
            print(f"      ... and {len(schema) - 10} more columns")

    conn.close()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if total_images > 0:
        print(f"✅ Images: {total_images:,}")
    else:
        print("⚠️  No 'images' table found")

    if total_detections > 0:
        print(f"✅ Detections: {total_detections:,}")
    else:
        print("⚠️  No detections table found")

    # Check if ready for Phase 2
    print("\n" + "=" * 70)
    print("PHASE 2 READINESS CHECK")
    print("=" * 70)

    ready = True

    if total_images < 100:
        print("❌ Not enough images (need at least 100, have {total_images})")
        ready = False
    else:
        print(f"✅ Sufficient images ({total_images:,})")

    if total_detections < 100:
        print(f"❌ Not enough detections (need at least 100, have {total_detections})")
        ready = False
    else:
        print(f"✅ Sufficient detections ({total_detections:,})")

    if ready:
        print("\n" + "=" * 70)
        print("🎉 DATABASE IS READY FOR PHASE 2 TRAINING!")
        print("=" * 70)
        print("\nNext step:")
        print("  python scripts/run_phase2_training.py \\")
        print(f"      --db {db_path} \\")
        print("      --output ./phase2_outputs")
    else:
        print("\n❌ Database not ready for Phase 2")
        print("   You may need to run Phase 1 data collection first")

except Exception as e:
    print(f"\n❌ Error reading database: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
