# Clasificación de Sentimiento de Noticias

## Descripción

### 1. Contexto del problema

El proyecto consiste en construir un sistema de clasificación de sentimiento para textos financieros.

Dado un texto como una frase, titular o fragmento de noticia financiera, el sistema deberá clasificarlo en una de tres categorías:

- **positive**: el texto comunica información favorable para una empresa, mercado o activo financiero.
- **neutral**: el texto comunica información descriptiva o factual, sin una dirección claramente positiva o negativa.
- **negative**: el texto comunica información desfavorable, como caídas, pérdidas, riesgos o deterioro en resultados.

| Texto | Sentimiento esperado |
|---|---|
| The company reported higher operating profit this quarter. | positive |
| The company announced the date of its annual meeting. | neutral |
| Sales declined compared with the previous year. | negative |

El objetivo final será comparar al menos dos enfoques de modelado:

1. TF-IDF + Logistic Regression.
2. Embeddings de Hugging Face + clasificador posterior.

---

## Instalación

### 1. Clonar repositorio

```bash
git clone <repo>
cd modulo1
```

### 2. Instalar dependencias

```bash
poetry install
```

### 3. Descargar datos con DVC

```bash
poetry run dvc pull
```

---

## Uso

### Ejecutar pruebas

```bash
poetry run pytest
```

### Construir contenedor Docker

```bash
docker build -t modulo-uno .
```

### Ejecutar contenedor

```bash
docker run --rm modulo-uno
```

---

## Estructura del proyecto

```text
modulo1/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
├── src/
│   └── proyecto/
│
├── tests/
├── Dockerfile
├── pyproject.toml
├── poetry.lock
├── README.md
└── .dvc/
```

---

## Datos

Se utilizó `sentences_50agree` para maximizar el tamaño del dataset, aceptando que puede contener más ruido en las etiquetas que versiones con mayor acuerdo entre anotadores.

### Fuente de datos

El dataset utilizado es **Financial PhraseBank**, un corpus de frases financieras anotadas con sentimiento.

Las categorías utilizadas son:

- positive
- neutral
- negative

El dataset se descarga desde Hugging Face y se almacena en:

```text
data/raw/financial_phrasebank.csv
```

El archivo raw no debe modificarse manualmente. Las transformaciones deben guardarse en `data/processed/`.

---

## Tecnologías utilizadas

- Python
- Poetry
- Pandas
- Scikit-learn
- Hugging Face
- DVC
- Docker
- Pytest
- MLflow

---

## Contacto

@author: bargof

- GitHub: github.com/bargof
- Email: 
