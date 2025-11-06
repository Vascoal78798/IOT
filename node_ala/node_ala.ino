/**
 * Nó da ala – Contagem de entrada/saída (Secção 2.1)
 *
 * Hardware alvo: Arduino UNO R4 WiFi
 * Responsabilidade: contar veículos que entram e saem por sensores ultrassónicos
 * dedicados, aplicar debounce e enviar eventos em formato JSON via Serial.
 */

#include <Arduino.h>
#include <cstring>

// Declaração antecipada para permitir chamadas antes da definição
void enviarEstadoResumo(bool forcar = false);

// ---- Configuração de hardware ----
constexpr uint8_t ENTRADA_ANALOG_PIN = A0;
constexpr uint8_t SAIDA_ANALOG_PIN = A1;
constexpr uint8_t QUALIDADE_AR_PIN = A2;          // Sensor MQ-7 (SEN0307)
constexpr uint8_t VENTOINHA_PWM_PIN = 9;          // Pino PWM para ventoinha

// ---- Parâmetros de deteção ----
constexpr int ANALOG_THRESHOLD = 30;              // Limite para considerar veículo próximo (0-1023)
constexpr unsigned long DEBOUNCE_INTERVAL_MS = 150;  // Bloqueio após deteção
constexpr uint8_t QUALIDADE_AR_AMOSTRAS = 10;
constexpr unsigned long QUALIDADE_AR_INTERVALO_MS = 2000;  // enviar a cada 2 s
constexpr float QUALIDADE_AR_REF_VOLT = 5.0f;
constexpr unsigned long ESTADO_INTERVALO_MS = 2000;
constexpr int ALERTA_DELTA_MAX = 3;
constexpr unsigned long ALERTA_PERSIST_MS = 5000;

// ---- Estruturas auxiliares ----
struct SensorUltrassonico {
  const char *id;  // "entrada" ou "saida"
  uint8_t analogPin;
  unsigned long ultimoDisparoMs;
  bool ativo;
};

// ---- Estado global ----
volatile int ocupacaoAla = 0;
unsigned long ultimaAtualizacaoMs = 0;
float qualidadeArBruto = 0.0f;
float qualidadeArTensao = 0.0f;
unsigned long ultimaLeituraQualidadeMs = 0;
int ventoinhaPercentual = 0;
int somaLugaresOcupados = 0;
bool alertaSensor = false;
unsigned long diferencaDesdeMs = 0;
unsigned long ultimoEnvioEstadoMs = 0;
unsigned long ultimaRececaoLugaresMs = 0;
SensorUltrassonico sensorEntrada{"entrada", ENTRADA_ANALOG_PIN, 0, false};
SensorUltrassonico sensorSaida{"saida", SAIDA_ANALOG_PIN, 0, false};

// ---- Funções utilitárias ----
bool detetarVeiculo(SensorUltrassonico &sensor) {
  int leitura = analogRead(sensor.analogPin);
  unsigned long agora = millis();

  if (leitura < ANALOG_THRESHOLD && !sensor.ativo &&
      (agora - sensor.ultimoDisparoMs) >= DEBOUNCE_INTERVAL_MS) {
    sensor.ativo = true;
    sensor.ultimoDisparoMs = agora;
    return true;
  }

  if (leitura >= ANALOG_THRESHOLD && sensor.ativo) {
    sensor.ativo = false;
  }

  return false;
}

// ---- 2.6 Envio de dados para o Pi (Serial JSON) ----
// Eventos simples de entrada/saída com total acumulado
void enviarEvento(const char *evento, int total) {
  Serial.print('{');
  Serial.print("\"evento\":\"");
  Serial.print(evento);
  Serial.print("\",\"total\":");
  Serial.print(total);
  Serial.print(",\"timestamp_ms\":");
  Serial.print(millis());
  Serial.println('}');
}

// Envia amostras de qualidade do ar (valor relativo + tensão calculada)
void enviarQualidadeAr(float valorBruto, float tensao) {
  Serial.print('{');
  Serial.print("\"evento\":\"qualidade_ar\",");
  Serial.print("\"valor_bruto\":");
  Serial.print(valorBruto, 1);
  Serial.print(",\"tensao_v\":");
  Serial.print(tensao, 3);
  Serial.print(",\"timestamp_ms\":");
  Serial.print(millis());
  Serial.println('}');
}

// Confirma ao Pi o estado atual da ventoinha (percentagem)
void enviarEstadoVentoinha() {
  Serial.print('{');
  Serial.print("\"evento\":\"ventoinha\",");
  Serial.print("\"percent\":");
  Serial.print(ventoinhaPercentual);
  Serial.print(",\"timestamp_ms\":");
  Serial.print(millis());
  Serial.println('}');
}

void atualizarQualidadeAr() {
  unsigned long agora = millis();
  if (agora - ultimaLeituraQualidadeMs < QUALIDADE_AR_INTERVALO_MS) {
    return;
  }

  long soma = 0;
  for (uint8_t i = 0; i < QUALIDADE_AR_AMOSTRAS; ++i) {
    soma += analogRead(QUALIDADE_AR_PIN);
    delay(5);
  }

  qualidadeArBruto = static_cast<float>(soma) / QUALIDADE_AR_AMOSTRAS;
  qualidadeArTensao = (qualidadeArBruto / 1023.0f) * QUALIDADE_AR_REF_VOLT;
  ultimaLeituraQualidadeMs = agora;

  enviarQualidadeAr(qualidadeArBruto, qualidadeArTensao);
  enviarEstadoResumo(false);
}

// ---- 2.4 Ventoinha: recebe % do Pi, aplica via PWM e confirma estado ----
// Lê comandos JSON do Pi: "percent" (0..100) e "lugares_ocupados"
void atualizarVentoinha() {
  bool alterou = false;
  while (Serial.available() > 0) {
    String linha = Serial.readStringUntil('\n');
    linha.trim();
    if (linha.isEmpty()) {
      continue;
    }

    int novoPercentual;
    if (extrairInteiro(linha, "\"percent\":", novoPercentual)) {
      ventoinhaPercentual = constrain(novoPercentual, 0, 100);
      aplicarPWMVentoinha();
      alterou = true;
    }

    int novaSoma;
    if (extrairInteiro(linha, "\"lugares_ocupados\":", novaSoma)) {
      somaLugaresOcupados = max(0, novaSoma);
      ultimaRececaoLugaresMs = millis();
      alterou = true;
    }
  }

  if (alterou) {
    enviarEstadoVentoinha();
    enviarEstadoResumo(true);
  }
}

void processarEntrada() {
  if (detetarVeiculo(sensorEntrada)) {
    ++ocupacaoAla;
    ultimaAtualizacaoMs = millis();
    enviarEvento("entrada", ocupacaoAla);
    enviarEstadoResumo(true);
  }
}

void processarSaida() {
  if (detetarVeiculo(sensorSaida)) {
    if (ocupacaoAla > 0) {
      --ocupacaoAla;
      ultimaAtualizacaoMs = millis();
      enviarEvento("saida", ocupacaoAla);
      enviarEstadoResumo(true);
    }
  }
}

// ---- Setup / Loop ----
void setup() {
  analogReadResolution(10);  // trabalhar numa escala 0-1023 semelhante ao exemplo

  Serial.begin(115200);
  while (!Serial) {
    ;
  }

  ocupacaoAla = 0;
  ultimaAtualizacaoMs = millis();
  sensorEntrada.ultimoDisparoMs = 0;
  sensorSaida.ultimoDisparoMs = 0;
  sensorEntrada.ativo = false;
  sensorSaida.ativo = false;

  pinMode(VENTOINHA_PWM_PIN, OUTPUT);
#if defined(ARDUINO_ARCH_RENESAS)
  analogWriteResolution(8);
#endif
  ventoinhaPercentual = 0;
  aplicarPWMVentoinha();

  Serial.println(F("[NODO-ALA] Sistema de contagem inicializado"));
  enviarEvento("estado_inicial", ocupacaoAla);
  enviarEstadoVentoinha();
}

void loop() {
  processarEntrada();
  processarSaida();
  atualizarQualidadeAr();
  atualizarVentoinha();
  atualizarAlertaSensor();
  enviarEstadoResumo(false);

  delay(50);
}

bool extrairInteiro(const String &linha, const char *chave, int &valor) {
  int idx = linha.indexOf(chave);
  if (idx < 0) {
    return false;
  }
  idx += strlen(chave);
  int fim = linha.indexOf(',', idx);
  if (fim < 0) {
    fim = linha.indexOf('}', idx);
  }
  if (fim < 0) {
    fim = linha.length();
  }
  valor = linha.substring(idx, fim).toInt();
  return true;
}

// Aplica o duty-cycle PWM correspondente à percentagem da ventoinha
void aplicarPWMVentoinha() {
  int duty = map(ventoinhaPercentual, 0, 100, 0, 255);
  analogWrite(VENTOINHA_PWM_PIN, duty);
}

// Envia estado agregado para o Pi (ocupação, ar, ventoinha, alerta)
void enviarEstadoResumo(bool forcar) {
  unsigned long agora = millis();
  if (!forcar && (agora - ultimoEnvioEstadoMs) < ESTADO_INTERVALO_MS) {
    return;
  }
  ultimoEnvioEstadoMs = agora;

  Serial.print('{');
  Serial.print("\"evento\":\"estado\",");
  Serial.print("\"ocupacao_ala\":");
  Serial.print(ocupacaoAla);
  Serial.print(",\"soma_lugares\":");
  Serial.print(somaLugaresOcupados);
  Serial.print(",\"qualidade_ar_bruto\":");
  Serial.print(qualidadeArBruto, 1);
  Serial.print(",\"qualidade_ar_tensao\":");
  Serial.print(qualidadeArTensao, 3);
  Serial.print(",\"ventoinha_percent\":");
  Serial.print(ventoinhaPercentual);
  Serial.print(",\"alerta_sensor\":");
  Serial.print(alertaSensor ? "true" : "false");
  Serial.print(",\"timestamp_ms\":");
  Serial.print(agora);
  Serial.println('}');
}

// ---- 2.5 Validação cruzada da ocupação ----
// Compara contador da ala (entrada/saída) com soma dos lugares do Pi.
// Se a diferença persistir acima de ALERTA_DELTA_MAX por ALERTA_PERSIST_MS, ativa alerta.
void atualizarAlertaSensor() {
  unsigned long agora = millis();
  int delta = abs(ocupacaoAla - somaLugaresOcupados);

  if (delta > ALERTA_DELTA_MAX) {
    if (diferencaDesdeMs == 0) {
      diferencaDesdeMs = agora;
    }
    if (!alertaSensor && (agora - diferencaDesdeMs) >= ALERTA_PERSIST_MS) {
      alertaSensor = true;
      enviarEstadoResumo(true);
    }
  } else {
    diferencaDesdeMs = 0;
    if (alertaSensor) {
      alertaSensor = false;
      enviarEstadoResumo(true);
    }
  }
}

