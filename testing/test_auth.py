from auth import verify_password, get_password_hash

def test_password_hashing():
    password = "admin_password"
    hashed = get_password_hash(password)
    
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrong_password", hashed) is False