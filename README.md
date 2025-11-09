# Sistema de Estacionamento Inteligente por Alas


## 1. Contexto / Problema

O objetivo do projeto é desenvolver um sistema de estacionamento inteligente organizado por alas (A, B, C, ...), que consiga:

- Dizer em tempo real que lugares estão livres e quais estão ocupados.
- Contar quantos carros entram e saem em cada ala, para confirmar se os sensores dos lugares estão a reportar bem.
- Mostrar tudo num dashboard (interface visual) hospedado na cloud.
- Prever a ocupação nas próximas horas (TinyML no edge + reforço cloud), com base em padrões reais de uso.
- Controlar a ventilação (fan) de cada ala de forma antecipada, para manter o ar aceitável quando há muita gente / muitos carros.
- Sincronizar dados para a cloud para histórico, analítica e comandos remotos.

A ideia não é só “um sensor num lugar e acabou”. É um sistema IoT completo, com vários níveis:
- Nós com sensores e atuadores (Arduino).
- Gateway que recebe dados e toma decisões (Raspberry Pi).
- Lógica de previsão.
- Dashboard para utilizador final e para administrador.

Este sistema aproxima-se de algo que podia ser usado num parque real: otimiza ocupação, dá histórico para gestão e ajuda a manter o ar respirável.

---

## 2. Arquitetura Geral

### 2.1 Nó de Lugar
Cada lugar tem:
- 1 sensor ultrassónico apontado à zona do carro.
- LEDs (verde/vermelho) para dizer se o lugar está livre ou ocupado.

Função:
- Mede distância.
- Decide localmente se está “livre” ou “ocupado” (com histerese).
- Acende o LED certo.
- Envia esse estado para o gateway.

**Corre em:** Arduino UNO R4 WiFi  
**Linguagem:** C/C++ (Arduino)

---

### 2.2 Nó de Ala
Cada ala tem o seu próprio nó “chefe”, que faz mais coisas:

- **Contagem da ocupação da ala:**  
  - A ala tem uma ENTRADA física e uma SAÍDA física.  
  - Na ENTRADA há um sensor dedicado (por exemplo ultrassónico A0).  
  - Na SAÍDA há outro sensor dedicado (ultrassónico A1).  
  - Quando o sensor da ENTRADA deteta um carro: `ocupacao_ala++`.  
  - Quando o sensor da SAÍDA deteta um carro: `ocupacao_ala--` (com limite mínimo 0).  
  - Para evitar contar um carro duas vezes por ruído, o nó aplica debounce (ignorar repetições durante alguns ms).

- **Sensor de ar:**  
  - Um sensor de qualidade do ar (ex.: MQ-7 para CO) mede “nível de ar sujo” na ala.

- **Ventoinha / fan:**  
  - A ventoinha é ligada ao Arduino e controlada por PWM (pode rodar mais rápido ou mais devagar conforme pedido).

- **Validação de sensores:**  
  - O nó de cada lugar diz se está ocupado ou livre.  
  - O nó da ala sabe `ocupacao_ala` pela contagem entrada/saída.  
  - Se a soma dos lugares ocupados não bate com `ocupacao_ala` durante algum tempo → gera `alerta_sensor` (possível falha de um sensor de lugar).

O nó da ala envia para o Raspberry Pi:
- `ocupacao_ala` (quantos carros dentro)
- qualidade do ar
- % da ventoinha atual
- alerta_sensor (sim/não)

**Corre em:** Arduino UNO R4 WiFi  
**Linguagem:** C/C++ (Arduino)

---

### 2.3 Gateway (Raspberry Pi)
O Raspberry Pi continua a ser o hub edge e faz a ponte para a cloud.

Funções principais:
- Recebe dados de todos os Arduinos (via MQTT ou série) e valida a ocupação localmente.
- Guarda histórico recente (SQLite/InfluxDB) para redundância.
- Agrega as métricas por ala e sincroniza lotes com a cloud (MQTT/HTTP seguro).
- Corre inferência de um modelo TinyML (treinado no Tiny Machine Learning Kit) para prever ocupação e antecipar ventilação.
- Aplica regras de segurança imediatas:
  - qualidade do ar muito má → ventoinha = 100%
  - qualidade do ar boa → ventoinha = mínimo
- Recebe da cloud recomendações/overrides (ex.: `percent`, `lugares_ocupados`) e envia comandos para o nó da ala.

**Corre em:** Raspberry Pi  
**Linguagem:** Python (MQTT, tflite-runtime, SQLite/InfluxDB)

---

### 2.4 Camada Cloud
A cloud fornece armazenamento durável, analítica e dashboards.

Responsabilidades:
- Ingestão de dados provenientes do Pi (MQTT IoT Core / HTTP).
- Persistência em base de dados gerida (ex.: DynamoDB/Timestream, Cosmos DB, Firestore...).
- Serviços de analítica/ML adicionais (SageMaker, Azure ML, Vertex AI) caso seja necessário re-treinar modelos mais pesados.
- Exposição de APIs para enviar comandos de volta ao Pi ou disponibilizar dados ao dashboard.

---

### 2.5 Dashboard

O dashboard passa a viver na cloud (Power BI, Grafana Cloud, Looker Studio ou web app dedicado).

Mostra em tempo real:
- Lugares livres / ocupados por ala.
- `ocupacao_ala` vs `soma_lugares` e `alerta_sensor`.
- Qualidade do ar e `ventoinha_percent` atual.
- Indicadores “vai encher” / “vai aliviar” com base no TinyML + previsões cloud.
- Alertas operacionais (ex.: “Ala A quase cheia → abrir Ala B?”).

Também disponibiliza uma vista simples para utilizador normal com os lugares ainda livres.

---

## 3. Comunicação e Mensagens

Os nós (Arduinos) comunicam com o Raspberry Pi usando MQTT (ou série, na fase de protótipo). O Pi agrega e repassa os dados para a cloud, mantendo dois fluxos principais:

1. **Edge → Cloud:** Pacotes JSON com métricas por ala/lugar (ocupação, qualidade do ar, ventoinha, alerta) enviados de forma batched para a cloud.
2. **Cloud → Edge:** Comandos ou recomendações (ex.: `{"percent":70,"lugares_ocupados":12}`) que o Pi encaminha para o nó da ala.

Formato típico (edge → cloud):

```json
{
  "ala": "A",
  "lugar": "A-03",
  "estado_lugar": "ocupado",
  "ocupacao_ala": 12,
  "soma_lugares": 11,
  "qualidade_ar": 410,
  "ventoinha_percent": 60,
  "alerta_sensor": false,
  "timestamp": "2025-10-30T12:15:00Z"
}
```

---

## 4. “Inteligência” do Sistema

### 4.1 Validação cruzada da ocupação
- A ala sabe `ocupacao_ala` porque tem um sensor na ENTRADA (contador++) e um sensor na SAÍDA (contador--).
- Em paralelo, cada lugar diz se está ocupado ou livre.
- O sistema compara:
  - `ocupacao_ala` (contagem entrada/saída)
  - vs soma dos lugares ocupados.
- Se a diferença for grande durante algum tempo → `alerta_sensor = true`.

Isto permite detetar falhas de sensores dos lugares sem ter de ir fisicamente verificar.

### 4.2 Previsão de ocupação
- O dataset é preparado no Pi e sincronizado com a cloud.
- O treino inicial é feito no Tiny Machine Learning Kit (TensorFlow Lite Micro), gerando um modelo `.tflite` que é colocado no Pi.
- O Pi corre inferência local a cada ciclo (tflite-runtime) usando hora/dia, ocupação média e fluxo recente.
- Opcionalmente, serviços cloud podem re-treinar modelos mais pesados e enviar novos parâmetros para o Pi.

### 4.3 Ventilação antecipada (feed-forward)
1. O Pi combina o resultado TinyML com regras de segurança e com recomendações vindas da cloud.
2. Se prevê que a ala vai encher → sobe a ventilação antes do ar degradar.
3. Se prevê que vai acalmar → baixa para o mínimo.
4. A cloud pode enviar overrides (ex.: eventos de manutenção).
5. Segurança local tem sempre prioridade (ar muito mau → 100%).

### 4.4 Analítica na Cloud
- Dashboards e relatórios em tempo real sobre ocupação, qualidade do ar e alertas.
- Comparação entre previsão local (TinyML) e previsões cloud.
- Histórico longo prazo para decisões de expansão do parque.

---

## 5. Resumo das Linguagens e Onde Corre

| Parte                              | Corre onde          | Linguagem / Ferramentas                              |
|-----------------------------------|---------------------|------------------------------------------------------|
| Nó do lugar (sensor + LED)        | Arduino UNO R4 WiFi | C/C++ (Arduino IDE / PlatformIO)                     |
| Nó da ala (entrada/saída/ar/fan)  | Arduino UNO R4 WiFi | C/C++ (Arduino)                                      |
| Gateway / previsão / lógica       | Raspberry Pi        | Python, MQTT, tflite-runtime, SQLite/InfluxDB        |
| Comunicação MQTT                  | Raspberry Pi / Cloud| Mosquitto, IoT Core (TLS), bibliotecas MQTT          |
| Cloud (armazenamento/analítica)   | AWS/Azure/GCP       | IoT Core + DynamoDB/Timestream/Cosmos/etc., Python   |
| Dashboard                         | Cloud (BI/Web app)  | Grafana/Power BI/Looker Studio ou web app (JS/Python)|
| Treino TinyML                     | Tiny ML Kit / PC    | TensorFlow Lite Micro, scripts de preparação de dados|

---

## 6. O que vamos demonstrar

No final queremos conseguir mostrar:

1. Um lugar que deteta se está ocupado ou livre, acende LED e envia estado.
2. Uma ala que tem ENTRADA e SAÍDA físicas separadas e mantém `ocupacao_ala` com ++ / --.
3. O sistema a detetar diferença entre `ocupacao_ala` e a soma dos lugares ocupados, e gerar `alerta_sensor`.
4. O Raspberry Pi a prever que a ala vai encher (modelo TinyML treinado no Tiny Machine Learning Kit) e a mandar subir a ventoinha antes do ar ficar mau, combinando regras de segurança.
5. Sincronização de dados com a cloud, dashboard cloud em tempo real e capacidade de enviar comandos de volta (override de ventoinha, alertas).

Isto junta eletrónica, firmware, comunicação, previsão e interface — um projeto IoT completo.