# Reconhecimento de Língua Gestual Portuguesa (LGP)

Aplicação Android que reconhece gestos da **Língua Gestual Portuguesa** em tempo real, a partir da câmara frontal, e os mostra como texto. O objetivo é facilitar a comunicação entre pessoas surdas e ouvintes. Todo o processamento é feito no dispositivo.

Projeto desenvolvido por **Ivanilson Braga**, **Zakhar Khomyakivskyy** e **Ektiandro Elizabeth** no âmbito da unidade curricular de Laboratório de Projeto.

## Gestos suportados

Água · Bom dia · Não · Olá · Por favor · Sim

## Como funciona

```
câmara ──► MediaPipe Hand Landmarker ──► 21 marcos (x, y, z) ──► normalização ──► rede neural ──► média de 10 frames ──► texto
```

1. **Marcos da mão.** O [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) deteta a mão e devolve 21 pontos 3D. O modelo nunca vê píxeis, o que o torna pequeno e rápido.
2. **Normalização.** Os pontos passam a ser relativos ao pulso e são divididos pelo tamanho da mão. Assim, a posição da mão no ecrã, a distância à câmara e o formato da imagem deixam de importar. A mesma função existe em Python (`ml/landmarks.py`) e em Kotlin, e há um teste que garante que dão o mesmo resultado.
3. **Classificação.** Uma rede densa pequena (17 mil parâmetros, 68 KB em TFLite) classifica a forma da mão em cada frame.
4. **Suavização.** A app faz a média das probabilidades das últimas 10 frames e só mostra o gesto se a confiança média for de pelo menos 60%.

## Resultados

Avaliação com validação cruzada de 5 blocos. Cada bloco é um troço contíguo das imagens de cada gesto, para que imagens quase iguais não apareçam ao mesmo tempo no treino e na validação.

| | Exatidão | F1 macro |
|---|---:|---:|
| Pipeline original (LSTM sobre coordenadas brutas) | 51.5% | 0.55 |
| Modelo atual, frame a frame | 68.9% | 0.67 |
| **Modelo atual, média de 10 frames (como na app)** | **85.9%** | **0.85** |

Com o limiar de 60% usado na app, 59% das janelas produzem uma resposta e todas as respostas dadas na validação estavam corretas.

| Gesto | Precisão | Recall | F1 |
|---|---:|---:|---:|
| Água | 0.83 | 0.89 | 0.86 |
| Bom dia | 0.82 | 1.00 | 0.90 |
| Não | 0.71 | 0.85 | 0.77 |
| Olá | 1.00 | 1.00 | 1.00 |
| Por favor | 0.94 | 0.63 | 0.76 |
| Sim | 0.87 | 0.73 | 0.79 |

As confusões mais comuns são *por favor* → *bom dia* e *sim* → *não*. Os resultados completos (incluindo a matriz de confusão) estão em `ml/training_results.json`.

## Estrutura

```
.
├── ml/                           Dados e treino (Python)
│   ├── dataset_limpo/            738 imagens, uma pasta por gesto
│   ├── landmarks.py              Normalização dos marcos (partilhada com a app)
│   ├── processar_dados.py        1. Extrai os marcos das imagens
│   ├── treinar_modelo.py         2. Avalia e treina o modelo
│   ├── converter_tflite.py       3. Converte para TFLite e copia para a app
│   ├── modelo_gestos_lgp.keras   Modelo treinado
│   ├── classes.json
│   ├── preprocessing_stats.json
│   ├── training_results.json
│   └── requirements.txt
└── android/                      Aplicação (Kotlin, Jetpack Compose, CameraX)
    └── app/src/main/assets/      hand_landmarker.task, modelo_gestos_lgp.tflite, classes.json
```

## Treinar o modelo

Requer **Python 3.11**.

```bash
cd ml
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python processar_dados.py   # marcos das imagens -> dados_processados.npz
python treinar_modelo.py    # validação cruzada + modelo final -> modelo_gestos_lgp.keras
python converter_tflite.py  # .tflite -> android/app/src/main/assets/
```

O `processar_dados.py` usa o mesmo `hand_landmarker.task` que a app. Muitas imagens do dataset são recortes justos da mão, onde o detetor de palmas falha. Por isso cada imagem é testada com várias margens e espelhada, o que permite extrair marcos de 83% das imagens (contra 38% no pipeline original).

## Aplicação Android

1. Abrir a pasta `android/` no Android Studio e aguardar o Gradle Sync.
2. Ligar um telemóvel com a depuração USB ativa (a app precisa da câmara frontal).
3. Executar a configuração `app` (`Shift + F10`).

Requisitos: Android 7.0 (API 24) ou superior.

Os testes unitários correm com `./gradlew testDebugUnitTest`.

## Limitações

- O dataset é pequeno: 123 imagens por gesto, gravadas por poucas pessoas.
- Os gestos são reconhecidos pela **forma da mão**, não pelo movimento. Gestos com a mesma forma e movimentos diferentes não se distinguem.
- A app usa apenas uma mão.

## Autores

- Ivanilson Braga
- Zakhar Khomyakivskyy
- Ektiandro Elizabeth

## Licença

Projeto académico. Qualquer reutilização, parcial ou total, deve referenciar os autores.
