# Clasificación de Sentimineto de Noticias

## Descripción

### 1. Contexto del problema

El proyecto consiste en construir un sistema de clasificación de sentimiento para textos financieros.

Dado un texto como una frase, titular o fragmento de noticia financiera, el sistema deberá clasificarlo en una de tres categorías:
- **positive**: el texto comunica información favorable para una empresa, mercado o activo financiero.
- **neutral**: el texto comunica información descriptiva o factual, sin una dirección claramente positiva o negativa.
- **negative**: el texto comunica información desfavorable, como caídas, pérdidas, riesgos o deterioro en resultados.

Ejemplo conceptual:

| Texto | Sentimiento esperado |
|---|---|
| The company reported higher operating profit this quarter. | positive |
| The company announced the date of its annual meeting. | neutral |
| Sales declined compared with the previous year. | negative |

El objetivo final será comparar al menos dos enfoques de modelado:

1. Un modelo baseline clásico basado en **TF-IDF + Logistic Regression**.
2. Un modelo basado en **embeddings de Hugging Face** y clasificación posterior.

## Instalación

## Uso

## Estructura del proyecto

## Datos

Se utilizó sentences_50agree para maximizar el tamaño del dataset, aceptando que puede contener más ruido en las etiquetas que versiones con mayor acuerdo entre anotadores.

## 2. Fuente de datos

El dataset utilizado es **Financial PhraseBank**, un corpus de frases financieras anotadas con sentimiento.

Este dataset contiene frases extraídas de noticias financieras y clasificadas en tres categorías principales:

- positive
- neutral
- negative

Para este proyecto se utilizará una versión del dataset descargada desde Hugging Face y guardada localmente en:

`data/raw/financial_phrasebank.csv`

El archivo raw no debe modificarse manualmente. Cualquier limpieza o transformación posterior deberá generar nuevos archivos dentro de `data/processed/`.


## Contacto

@author: bargof
* github: github.com/bargof
* email:
