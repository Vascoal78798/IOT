# Guia de Montagem para Projeto Arduino

Este guia descreve a montagem dos componentes na breadboard e as ligações com o Arduino, adaptado para um setup onde não se utilizam resistências para os LEDs.

## 1. Conhecendo a Breadboard

- Uma breadboard serve para montar circuitos eletrónicos sem solda.
- **Linhas laterais (rails):**
  - VERMELHA (+): Alimentação positiva.
  - AZUL/PRETA (-): Terra (GND).
  - São contínuas ao longo do comprimento (algumas breadboards têm cortes; neste projeto usamos apenas metade, se necessário).
- **Colunas centrais:** Grupos de 5 furos ligados verticalmente (A5, B5, C5, D5, E5 estão conectados; A5 NÃO está ligado a A6).
- Para este projeto usamos:
  - Rail VERMELHA → 5 V.
  - Rail AZUL → GND.
- Isto distribui 5 V e GND de forma organizada, evitando cabos diretamente no Arduino.

## 2. Ligando o Arduino à Breadboard (Alimentação)

1. **GND:** liga um cabo do pino GND do Arduino a qualquer furo da linha AZUL (-) da breadboard. Resultado: toda a linha azul torna-se terra (0 V).
2. **5 V:** liga um cabo do pino 5 V do Arduino a qualquer furo da linha VERMELHA (+) da breadboard. Resultado: toda a linha vermelha passa a ter +5 V.

> ⚠️ Nunca ligues diretamente a linha vermelha (+) à linha azul (-). Isso causa curto-circuito e pode danificar o Arduino.

## 3. Ligando os LEDs na Breadboard (sem resistências)

- Atenção: ligar LEDs diretamente sem resistência pode reduzir a vida útil e danificar LED ou pino. Montagem sob risco.
- O código usa:
  - LED verde → pino digital 9.
  - LED vermelho → pino digital 10.
- Cada LED tem polaridade:
  - Perna longa: positivo (ânodo).
  - Perna curta: negativo (cátodo).

### 3.1 LED verde (pino 9)

- Coloca o LED verde na breadboard, pernas em colunas diferentes.
- Perna curta → liga à linha AZUL (GND).
- Perna longa → liga ao pino digital 9 do Arduino.

**Resumo LED verde:** pino 9 → perna longa; perna curta → linha AZUL (GND).

### 3.2 LED vermelho (pino 10)

- Coloca o LED vermelho noutra zona, pernas em colunas diferentes.
- Perna curta → liga à linha AZUL (GND).
- Perna longa → liga ao pino digital 10 do Arduino.

**Resumo LED vermelho:** pino 10 → perna longa; perna curta → linha AZUL (GND).

## 4. Ligando o Sensor Ultrassónico (HC-SR04)

- O código usa `TRIGGER_PIN = 2` e `ECHO_PIN = 3`.
- O sensor possui pinos VCC, TRIG, ECHO, GND.
- Encaixa o sensor na breadboard com cada pino numa coluna diferente.
- Ligações:
  - VCC → linha VERMELHA (+5 V).
  - GND → linha AZUL (GND).
  - TRIG → pino digital 2 do Arduino.
  - ECHO → pino digital 3 do Arduino.
