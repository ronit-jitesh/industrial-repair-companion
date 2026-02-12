#!/usr/bin/env python3
"""
generate_repair_logs.py
Generates 10,000+ synthetic repair log records for Siemens SINAMICS VFD systems.
Simulates diverse environmental conditions, realistic error codes, and technician notes.
"""

import csv
import random
from datetime import datetime, timedelta

# ─── Configuration ───────────────────────────────────────────────────────────

NUM_RECORDS = 10500
OUTPUT_FILE = "data/repair_logs.csv"
RANDOM_SEED = 42

# ─── Reference Data ──────────────────────────────────────────────────────────

MACHINE_MODELS = [
    "SINAMICS-G120", "SINAMICS-G120C", "SINAMICS-G120D",
    "SINAMICS-V20", "SINAMICS-S120",
]

SITES = [
    "Houston Plant", "Detroit Assembly", "Munich Factory", "Shanghai Hub",
    "São Paulo Works", "Chennai Unit", "Düsseldorf Line", "Chicago South",
    "Johannesburg Site", "Melbourne Depot",
]

# Realistic Siemens fault codes with associated metadata
ERROR_PROFILES = {
    "F30001": {
        "desc": "Overcurrent",
        "notes": [
            "Short circuit detected on motor cable phase U",
            "Ground fault on output — insulation test failed at 0.3 MOhm",
            "Motor parameter mismatch — re-ran motor identification p1900",
            "Sudden load impact from conveyor jam, current spike to 220%",
            "Cable damage found at cable gland entry point",
            "Replaced motor cable — old cable had nicked insulation near junction box",
            "Motor winding fault confirmed — replaced motor",
        ],
        "temp_bias": 0, "vib_bias": 2.0,
    },
    "F30002": {
        "desc": "DC link overvoltage",
        "notes": [
            "Deceleration ramp too fast for load inertia — increased p1121 to 15s",
            "Braking resistor open circuit — replaced resistor",
            "Blocked cooling fan caused thermal runaway, DC link exceeded 810V",
            "Mains voltage swell measured at 445V (nominal 400V) during afternoon",
            "High inertia grinding wheel — installed braking module with 50 Ohm resistor",
            "Solar panel back-feed causing voltage swell on supply bus",
            "Replaced capacitor C2 on DC bus — ESR measured 3x above baseline",
            "Deceleration from 50Hz to 0Hz in 2s with 500kg flywheel — doubled ramp time",
            "Fan blockage confirmed — cleaned dust accumulation from heat sink fins",
            "Coupling failure caused sudden load release — motor overspeed to 2100rpm",
        ],
        "temp_bias": 8, "vib_bias": 1.0,
    },
    "F30003": {
        "desc": "DC link undervoltage",
        "notes": [
            "Mains brownout — UPS installed on control supply",
            "Supply contactor coil failure — replaced contactor",
            "Power cable undersized — voltage drop 18V over 80m run",
            "Mains fuse blown on phase L2 — replaced and checked upstream protection",
            "Unstable generator supply at remote site — added line reactor",
        ],
        "temp_bias": 0, "vib_bias": 0,
    },
    "F30004": {
        "desc": "Overtemperature heat sink",
        "notes": [
            "Cooling fan failed — bearing seized, replaced fan assembly",
            "Air filter clogged with cotton fibers — cleaned and replaced filter",
            "Cabinet ambient measured at 52°C — installed cabinet A/C unit",
            "Clearance above drive was only 30mm — reinstalled with 100mm gap",
            "Dust accumulation on heat sink fins — cleaned with compressed air",
            "Fan spinning but airflow weak — fan blade partially broken, replaced",
            "Drive location next to furnace — relocated to cooler section of cabinet",
        ],
        "temp_bias": 15, "vib_bias": 0.5,
    },
    "F30005": {
        "desc": "I2t overload",
        "notes": [
            "Motor stalling intermittently — mechanical binding in gearbox",
            "Drive undersized for application — upgraded from 7.5kW to 11kW",
            "Frequent start-stop cycles exceeding thermal capacity",
            "Bearing failure causing excessive friction — replaced motor bearings",
            "Belt tension too high on conveyor — adjusted tensioner",
        ],
        "temp_bias": 5, "vib_bias": 3.0,
    },
    "F30017": {
        "desc": "EEPROM data error",
        "notes": [
            "Power loss during parameter save — performed factory reset p0970=1",
            "Electrical noise spike during thunderstorm — installed surge protector",
            "Control unit hardware defect — replaced CU240E",
            "Firmware corruption after power dip — reloaded firmware v4.7 SP11",
        ],
        "temp_bias": 0, "vib_bias": 0,
    },
    "F30021": {
        "desc": "Ground fault",
        "notes": [
            "Motor cable insulation breakdown at 15m mark — replaced cable section",
            "Moisture ingress in motor terminal box — dried and sealed cable glands",
            "Motor winding insulation degraded to 0.5 MOhm — motor rewound",
            "Cable routing through wet area — rerouted through cable tray above",
            "Long cable run 75m without filter — installed dU/dt output filter",
        ],
        "temp_bias": 0, "vib_bias": 1.5,
    },
    "F07011": {
        "desc": "Motor overtemperature",
        "notes": [
            "External motor fan not running — contactor coil failed, replaced",
            "Motor running at 10Hz continuously without forced ventilation — added fan",
            "PTC sensor wiring intermittent — re-terminated at motor terminal box",
            "Motor ambient in enclosed space reached 55°C — improved ventilation",
            "Incorrect temperature sensor type selected — changed p0601 from KTY to PTC",
            "Motor overloaded at 115% for 3 hours — reduced production speed",
        ],
        "temp_bias": 12, "vib_bias": 1.0,
    },
    "F07900": {
        "desc": "Encoder signal error",
        "notes": [
            "Encoder cable shield not grounded — reconnected shield at drive end",
            "Encoder coupling loose — retightened set screws with Loctite",
            "Encoder cable routed next to VFD output cable — rerouted with 200mm separation",
            "Encoder supply voltage measured 4.2V at encoder (should be 5.0V) — cable too long",
            "Encoder replaced after motor swap — updated p0408 to match new PPR",
        ],
        "temp_bias": 0, "vib_bias": 4.0,
    },
    "A07910": {
        "desc": "Encoder signal warning",
        "notes": [
            "Speed deviation intermittent — encoder connector partially unseated",
            "Vibration at motor causing encoder jitter — installed vibration dampener",
            "EMC interference from nearby welding robot — improved cable shielding",
            "Encoder bearing worn — preemptive replacement scheduled",
        ],
        "temp_bias": 0, "vib_bias": 3.5,
    },
}

OUTCOMES = ["Fixed", "Pending", "Replaced_Unit", "Temporary_Fix"]
OUTCOME_WEIGHTS = [0.65, 0.10, 0.15, 0.10]

TECHNICIAN_IDS = [f"T-{str(i).zfill(4)}" for i in range(1, 51)]

TECHNICIAN_NAMES = [
    "J. Doe", "A. Smith", "M. Weber", "S. Patel", "R. García",
    "K. Tanaka", "L. Müller", "P. Johnson", "C. Silva", "D. Kim",
    "E. Brown", "F. Chen", "G. Anderson", "H. Nakamura", "I. Petrov",
    "N. Williams", "O. Schmidt", "Q. López", "T. Kumar", "U. Fischer",
    "V. Martinez", "W. Lee", "X. Taylor", "Y. Yamamoto", "Z. Ivanov",
    "A. Davis", "B. Schneider", "C. Hernandez", "D. Sato", "E. Wilson",
    "F. Becker", "G. Morales", "H. Suzuki", "I. Thomas", "J. Hoffmann",
    "K. Ramirez", "L. Watanabe", "M. Jackson", "N. Wagner", "O. Torres",
    "P. Takahashi", "Q. Harris", "R. Koch", "S. Flores", "T. Ito",
    "U. Robinson", "V. Bauer", "W. Rivera", "X. Kimura", "Y. Clark",
]

# ─── Generator Functions ─────────────────────────────────────────────────────


def random_date(start_year=2021, end_year=2024):
    """Generate a random date between start_year and end_year."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_machine_id(model):
    """Generate a unique-style machine ID."""
    unit_num = random.randint(1, 200)
    return f"{model}-{str(unit_num).zfill(3)}"


def generate_temp(error_code):
    """Generate operating temperature with realistic bias per error type."""
    base = random.gauss(45, 12)  # Normal industrial operating range
    bias = ERROR_PROFILES[error_code]["temp_bias"]
    temp = base + bias + random.uniform(-3, 3)
    temp = max(15.0, min(95.0, temp))  # Clamp to realistic range
    # ~10% chance of missing value
    if random.random() < 0.10:
        return None
    return round(temp, 1)


def generate_vibration(error_code):
    """Generate vibration level (0-10 scale) with bias per error type."""
    base = random.gauss(3.0, 1.5)
    bias = ERROR_PROFILES[error_code]["vib_bias"]
    vib = base + bias * random.uniform(0.3, 1.0)
    vib = max(0.1, min(10.0, vib))
    # ~8% chance of missing
    if random.random() < 0.08:
        return None
    return round(vib, 1)


def generate_humidity():
    """Generate humidity reading (20-95%)."""
    humidity = random.gauss(55, 18)
    humidity = max(20, min(95, humidity))
    # ~5% chance of missing
    if random.random() < 0.05:
        return None
    return round(humidity, 0)


def generate_record(record_id):
    """Generate a single repair log record."""
    error_code = random.choice(list(ERROR_PROFILES.keys()))
    profile = ERROR_PROFILES[error_code]
    model = random.choice(MACHINE_MODELS)
    tech_idx = random.randint(0, len(TECHNICIAN_IDS) - 1)

    return {
        "Log_ID": record_id,
        "Date": random_date().strftime("%Y-%m-%d"),
        "Machine_ID": generate_machine_id(model),
        "Error_Code": error_code,
        "Error_Description": profile["desc"],
        "Operating_Temp": generate_temp(error_code),
        "Vibration_Level": generate_vibration(error_code),
        "Humidity": generate_humidity(),
        "Technician_ID": TECHNICIAN_IDS[tech_idx],
        "Technician_Name": TECHNICIAN_NAMES[tech_idx],
        "Technician_Notes": random.choice(profile["notes"]),
        "Site_Location": random.choice(SITES),
        "Outcome": random.choices(OUTCOMES, weights=OUTCOME_WEIGHTS, k=1)[0],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)

    print(f"Generating {NUM_RECORDS} synthetic repair logs...")

    fieldnames = [
        "Log_ID", "Date", "Machine_ID", "Error_Code", "Error_Description",
        "Operating_Temp", "Vibration_Level", "Humidity",
        "Technician_ID", "Technician_Name", "Technician_Notes",
        "Site_Location", "Outcome",
    ]

    records = [generate_record(i + 1) for i in range(NUM_RECORDS)]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"✅ Generated {NUM_RECORDS} records → {OUTPUT_FILE}")
    print(f"{'='*60}")

    # Error code distribution
    code_counts = {}
    temp_filled = 0
    vib_filled = 0
    hum_filled = 0
    for r in records:
        code_counts[r["Error_Code"]] = code_counts.get(r["Error_Code"], 0) + 1
        if r["Operating_Temp"] is not None:
            temp_filled += 1
        if r["Vibration_Level"] is not None:
            vib_filled += 1
        if r["Humidity"] is not None:
            hum_filled += 1

    print("\n📊 Error Code Distribution:")
    for code in sorted(code_counts.keys()):
        desc = ERROR_PROFILES[code]["desc"]
        count = code_counts[code]
        pct = count / NUM_RECORDS * 100
        print(f"   {code} ({desc:.<30s}): {count:>5} ({pct:.1f}%)")

    print(f"\n📉 Missing Data (simulating real-world messiness):")
    print(f"   Operating_Temp:  {NUM_RECORDS - temp_filled} missing ({(NUM_RECORDS - temp_filled)/NUM_RECORDS*100:.1f}%)")
    print(f"   Vibration_Level: {NUM_RECORDS - vib_filled} missing ({(NUM_RECORDS - vib_filled)/NUM_RECORDS*100:.1f}%)")
    print(f"   Humidity:        {NUM_RECORDS - hum_filled} missing ({(NUM_RECORDS - hum_filled)/NUM_RECORDS*100:.1f}%)")

    print(f"\n🏭 Sites: {len(set(r['Site_Location'] for r in records))}")
    print(f"👷 Technicians: {len(set(r['Technician_ID'] for r in records))}")
    print(f"🔧 Machines: {len(set(r['Machine_ID'] for r in records))}")
    print(f"📅 Date Range: {min(r['Date'] for r in records)} → {max(r['Date'] for r in records)}")


if __name__ == "__main__":
    main()
