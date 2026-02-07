#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Clear all output tables in database"""

import psycopg

# Database connection
conn_str = 'host=localhost port=5432 dbname=test_db user=postgres password=123456'

# Tables to truncate
TABLES_TO_TRUNCATE = [
    # Module1 output tables
    'module1_output_orderlog',
    'module1_output_shipmentlog',
    'module1_output_cutlog',
    'module1_output_supplydemandlog',
    'module1_output_summary',

    # Module3 output tables
    'module3_output_netdemand',

    # Module4 output tables
    'module4_output_productionplan',
    'module4_output_capacityexceed',
    'module4_output_validation',
    'module4_output_changeoverlog',

    # Module5 output tables
    'module5_output_deploymentplan',
    'module5_output_unfulfilledlog',
    'module5_output_stockonhandlog',
    'module5_output_validation',

    # Module6 output tables
    'module6_output_deliveryplan',
    'module6_output_vehiclelog',
    'module6_output_truckusagelog',
    'module6_output_unsatisfiedmdqlog',
    'module6_output_validationlog',
    'module6_output_bypassrulehitlog',

    # Summary output tables
    'summary_output_ordershipmentcutsummary',
    'summary_output_fullchangeoverlog',
    'summary_output_fullcapacityexceed',
    'summary_output_fullproductionplan',
    'summary_output_fulldeploymentplan',
    'summary_output_fulldeliveryplan',
    'summary_output_fulltruckusage',

    # Orchestrator state tables
    'orchestrator_unrestricted_inventory',
    'orchestrator_open_deployment',
    'orchestrator_open_deployment_pastdue_cleanup',
    'orchestrator_planning_intransit',
    'orchestrator_space_quota',
    'orchestrator_delivery_gr',
    'orchestrator_production_gr',
    'orchestrator_production_plan_backlog',
    'orchestrator_shipment_log',
    'orchestrator_delivery_shipment_log',
    'orchestrator_inventory_change_log',
    'orchestrator_daily_logs',
]

def truncate_all_tables():
    """Truncate all output tables"""
    print("=" * 60)
    print("Truncating database output tables")
    print("=" * 60)

    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                truncated = 0
                skipped = 0

                for table in TABLES_TO_TRUNCATE:
                    try:
                        # Check if table exists
                        cur.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables
                                WHERE table_name = %s
                            )
                        """, (table,))
                        result = cur.fetchone()
                        exists = result[0] if result else False

                        if exists:
                            cur.execute(f'TRUNCATE TABLE {table} CASCADE')
                            print(f"[OK] Truncated: {table}")
                            truncated += 1
                        else:
                            print(f"[--] Skipped (not exists): {table}")
                            skipped += 1

                    except Exception as e:
                        print(f"[ERR] Error {table}: {e}")

                conn.commit()

        print("=" * 60)
        print(f"Done! Truncated {truncated} tables, skipped {skipped} tables")
        print("=" * 60)

    except Exception as e:
        print(f"Database connection error: {e}")

if __name__ == '__main__':
    truncate_all_tables()
