from web.auth import get_password_hash


def test_tc004_password_hashing_uniqueness():
    """Verify that hashing the same password twice yields different results."""
    password = "admin"
    hash_one = get_password_hash(password)
    hash_two = get_password_hash(password)

    assert hash_one != hash_two, "Hashes should be unique due to salt"
