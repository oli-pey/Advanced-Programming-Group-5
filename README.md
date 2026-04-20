# Advanced-Programming-Group-5

erDiagram
    USERS ||--o{ ENTRIES : "has many"
    
    USERS {
        INTEGER id PK
        VARCHAR(100) username UK "Index"
        VARCHAR(255) password_hash
        BOOLEAN is_admin
        DATETIME created_at
    }

    ENTRIES {
        INTEGER id PK "Index"
        INTEGER user_id FK "Index"
        BLOB original_image
        BLOB downsized_image
        TEXT prediction
        VARCHAR(50) model_name
        TEXT probability
        DATETIME created_at
    }