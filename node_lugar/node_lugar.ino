/**
 * Nó do lugar – Arduino UNO R4 WiFi
 *
 * Lê um sensor ultrassónico para determinar se o lugar está ocupado ou livre,
 * aplica histerese, controla LEDs verde/vermelho e envia o estado por serial
 * num formato JSON simples.
 */

#include <Arduino.h>

// ---- Configuração de hardware ----
constexpr uint8_t TRIGGER_PIN = 2;
constexpr uint8_t ECHO_PIN = 3;
constexpr uint8_t LED_VERDE_PIN = 9;
constexpr uint8_t LED_VERMELHO_PIN = 10;

// Identificação do lugar (ajustar por dispositivo)
constexpr const char *ID_LUGAR = "A-01";

// ---- Parâmetros de medição ----
constexpr uint16_t CALIBRATION_SAMPLES = 50;     // Leituras para referência a vazio
constexpr uint16_t SAMPLE_COUNT = 8;             // Leituras por ciclo (média)
// Margens relativas (percentagem da referência) com limites mínimos absolutos
constexpr float MARGEM_OCUPADO_FATOR = 0.4f;      // 40% da referência
constexpr float MARGEM_LIVRE_FATOR = 0.2f;        // 20% da referência
constexpr float MARGEM_OCUPADO_MIN_CM = 3.0f;     // mínimo 3 cm
constexpr float MARGEM_LIVRE_MIN_CM = 1.0f;       // mínimo 1 cm
constexpr unsigned long STATUS_INTERVAL_MS = 5000;  // Enviar estado a cada 5 s
constexpr unsigned long BETWEEN_SAMPLES_DELAY_MS = 40;

enum class EstadoLugar { Livre, Ocupado };

// ---- Variáveis globais ----
float referenciaDistanciaCm = 0.0f;
EstadoLugar estadoAtual = EstadoLugar::Livre;
unsigned long ultimoEnvioMs = 0;

// ---- Funções utilitárias ----

float medirDistanciaCm() {
  digitalWrite(TRIGGER_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIGGER_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGGER_PIN, LOW);

  // Timeout de ~30 ms (aprox. 5 m)
  unsigned long duracao = pulseIn(ECHO_PIN, HIGH, 30000UL);
  if (duracao == 0) {
    return -1.0f;  // Sem leitura válida
  }

  // Conversão: velocidade do som ~343 m/s → 58 µs por centímetro
  return static_cast<float>(duracao) / 58.0f;
}

float mediaDistancias(uint16_t amostras) {
  float soma = 0.0f;
  uint16_t validas = 0;

  for (uint16_t i = 0; i < amostras; ++i) {
    float leitura = medirDistanciaCm();
    if (leitura > 0.0f) {
      soma += leitura;
      ++validas;
    }
    delay(BETWEEN_SAMPLES_DELAY_MS);
  }

  if (validas == 0) {
    return -1.0f;
  }

  return soma / static_cast<float>(validas);
}

EstadoLugar determinarEstado(float distancia, EstadoLugar anterior) {
  if (distancia < 0.0f) {
    return anterior;
  }

  float margemOcupado = max(MARGEM_OCUPADO_FATOR * referenciaDistanciaCm,
                            MARGEM_OCUPADO_MIN_CM);
  float margemLivre = max(MARGEM_LIVRE_FATOR * referenciaDistanciaCm,
                          MARGEM_LIVRE_MIN_CM);

  float limiteOcupado = referenciaDistanciaCm - margemOcupado;
  float limiteLivre = referenciaDistanciaCm - margemLivre;

  if (distancia < limiteOcupado) {
    return EstadoLugar::Ocupado;
  }

  if (distancia > limiteLivre) {
    return EstadoLugar::Livre;
  }

  return anterior;
}

void atualizarLeds(EstadoLugar estado) {
  digitalWrite(LED_VERDE_PIN, estado == EstadoLugar::Livre ? HIGH : LOW);
  digitalWrite(LED_VERMELHO_PIN, estado == EstadoLugar::Ocupado ? HIGH : LOW);
}

void enviarEstado(EstadoLugar estado, float distanciaCm) {
  const unsigned long timestamp = millis();
  Serial.print('{');
  Serial.print("\"id_lugar\":\"");
  Serial.print(ID_LUGAR);
  Serial.print("\",\"estado_lugar\":\"");
  Serial.print(estado == EstadoLugar::Ocupado ? "ocupado" : "livre");
  Serial.print("\",\"distancia_cm\":");
  Serial.print(distanciaCm, 1);
  Serial.print(",\"referencia_cm\":");
  Serial.print(referenciaDistanciaCm, 1);
  Serial.print(",\"timestamp_ms\":");
  Serial.print(timestamp);
  Serial.println('}');
}

void calibrarReferencia() {
  Serial.println(F("[NODO-LUGAR] A calibrar distância de referência..."));

  float leitura = mediaDistancias(CALIBRATION_SAMPLES);
  while (leitura < 0.0f) {
    Serial.println(F("[NODO-LUGAR] Falha na calibração, a repetir."));
    delay(1000);
    leitura = mediaDistancias(CALIBRATION_SAMPLES);
  }

  referenciaDistanciaCm = leitura;
  Serial.print(F("[NODO-LUGAR] Referência calibrada: "));
  Serial.print(referenciaDistanciaCm, 1);
  Serial.println(F(" cm"));
}

// ---- Setup / Loop ----

void setup() {
  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(LED_VERDE_PIN, OUTPUT);
  pinMode(LED_VERMELHO_PIN, OUTPUT);

  digitalWrite(LED_VERDE_PIN, LOW);
  digitalWrite(LED_VERMELHO_PIN, LOW);

  Serial.begin(115200);
  while (!Serial) {
    ;
  }

  calibrarReferencia();
  estadoAtual = EstadoLugar::Livre;
  atualizarLeds(estadoAtual);
  enviarEstado(estadoAtual, referenciaDistanciaCm);
  ultimoEnvioMs = millis();
}

void loop() {
  float distanciaMedida = mediaDistancias(SAMPLE_COUNT);
  EstadoLugar novoEstado = determinarEstado(distanciaMedida, estadoAtual);

  if (novoEstado != estadoAtual) {
    estadoAtual = novoEstado;
    atualizarLeds(estadoAtual);
    enviarEstado(estadoAtual, distanciaMedida);
    ultimoEnvioMs = millis();
  } else if (millis() - ultimoEnvioMs >= STATUS_INTERVAL_MS) {
    enviarEstado(estadoAtual, distanciaMedida);
    ultimoEnvioMs = millis();
  }

  delay(100);
}

