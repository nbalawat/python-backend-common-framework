#!/usr/bin/env python3
"""Test first two fixed modules."""

import subprocess

def test_module(module_name, test_code):
    print(f"\nTesting {module_name}...")
    result = subprocess.run(
        ["uv", "run", "--no-project", "python", "-c", test_code],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ {module_name}: SUCCESS")
        print(result.stdout)
        return True
    else:
        print(f"❌ {module_name}: FAILED")
        print(result.stderr)
        return False

# Test commons-testing
testing_code = '''
import commons_testing
from commons_testing import AsyncTestCase, fake, DataGenerator

print("✓ commons-testing imported successfully")

# Test data generator
gen = DataGenerator(seed=42)
random_str = gen.random_string(10)
random_email = gen.random_email()
print(f"✓ DataGenerator: {random_str}, {random_email}")

# Test faker
fake_name = fake.name()
print(f"✓ Faker: {fake_name}")

print("✓ commons-testing basic functionality working")
'''

# Test commons-cloud
cloud_code = '''
import commons_cloud
from commons_cloud import CloudProvider, StorageClient, SecretManager

print("✓ commons-cloud imported successfully")

# Test cloud provider  
provider = CloudProvider("aws")
print(f"✓ CloudProvider created: {provider.name}")

# Test storage client
storage = StorageClient("s3://test-bucket")
print(f"✓ StorageClient created for: {storage.bucket}")

# Test secret manager
secrets = SecretManager("aws")
print(f"✓ SecretManager created for: {secrets.provider}")

print("✓ commons-cloud basic functionality working")
'''

success_count = 0
success_count += test_module("commons-testing", testing_code)
success_count += test_module("commons-cloud", cloud_code)

print(f"\n{success_count}/2 modules working")