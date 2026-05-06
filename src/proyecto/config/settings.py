from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# ═══════════════════════════════════════════
# Definición de la configuración con validación
# ═══════════════════════════════════════════


class Settings(BaseSettings):
    """
    Configuración centralizada de la aplicación.

    Carga automáticamente desde variables de entorno y archivos .env.
    Cada campo tiene validación de tipo y rango incorporada.
    """

    # --- Entorno ---
    environment: str = Field(
        default="development",
        description="Entorno de ejecución: development, staging, production",
    )
    debug: bool = Field(default=False, description="Modo debug activo")
    base_dir: Path = Path(__file__).resolve().parents[3]

    # --- Base de datos ---
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")

    # --- APIs externas ---
    bloomberg_api_key: str = Field(default="", description="API key de Bloomberg")
    max_retries: int = Field(
        default=3,
        ge=1,  # ge = greater than or equal (mínimo 1 reintento)
        le=10,  # le = less than or equal (máximo 10 para no abusar)
        description="Reintentos máximos para llamadas a APIs",
    )
    api_timeout: float = Field(
        default=30.0,
        gt=0,  # gt = greater than (debe ser positivo)
        description="Timeout en segundos para llamadas a APIs",
    )

    # --- Modelo de ML ---
    model_path: str = Field(default="models/latest.pkl", description="Ruta al modelo")
    random_seed: int = Field(default=42, description="Seed para reproducibilidad")
    batch_size: int = Field(default=32, gt=0, description="Tamaño de batch")

    # --- Logging ---
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    @property
    def log_file_path(self) -> Path:
        return self.base_dir / self.log_file

    # --- Configuración de pydantic-settings ---
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignora variables de entorno no definidas aquí
    }
    models_dir: Path = base_dir / "models"


# ═══════════════════════════════════════════
# Patrón singleton: una sola instancia de configuración
# ═══════════════════════════════════════════
def get_settings() -> Settings:
    """Retorna la configuración de la aplicación."""
    return Settings()


settings = get_settings()
