# Sistema de Estacionamento Inteligente por Alas


## 1. Contexto / Problema

O objetivo do projeto é desenvolver um sistema de estacionamento inteligente organizado por alas (A, B, C, ...), que consiga:

- Dizer em tempo real que lugares estão livres e quais estão ocupados.
- Contar quantos carros entram e saem em cada ala, para confirmar se os sensores dos lugares estão a reportar bem.
- Mostrar tudo num dashboard (interface visual).
- Prever a ocupação nas próximas horas, com base em padrões reais de uso.
- Controlar a ventilação (fan) de cada ala de forma antecipada, para manter o ar aceitável quando há muita gente / muitos carros.

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
O Raspberry Pi faz de cérebro central / servidor.

Funções:
- Recebe dados de todos os Arduinos (via série USB ou via MQTT).
- Guarda histórico localmente (por exemplo, SQLite ou InfluxDB).
- Faz previsão da ocupação de cada ala para os próximos minutos.  
  - Usa hora do dia, dia da semana, histórico recente e fluxo de entradas recentes (quantos carros entraram na ENTRADA da ala nos últimos minutos).
- Decide a velocidade base da ventoinha por ala de forma antecipada:
  - Se prevê que a ala vai encher, sobe a ventoinha antes do ar ficar mau.
  - Se prevê que a ala vai aliviar, baixa.

- Aplica regras de segurança:
  - Se o valor do sensor de ar estiver muito alto → ventoinha = 100%.
  - Se estiver limpo → ventoinha = mínimo (por ex. 20%).

- Compara também a ocupação reportada pelos lugares com a ocupação calculada pela contagem entrada/saída (`ocupacao_ala`) para validar se os sensores estão coerentes.

O Pi envia comandos de volta para o Arduino da ala, por exemplo “ventoinha = 60%”.

**Corre em:** Raspberry Pi  
**Linguagem:** Python  
**Tecnologias usadas:** Python scripts, MQTT broker (Mosquitto)

---

### 2.4 Dashboard
O dashboard mostra o estado do parque para duas pessoas diferentes:
- utilizador normal (onde posso estacionar agora?)
- administrador (isto está a encher, abro outra ala? qualidade do ar está ok?)

O dashboard apresenta:
- Lugares livres e ocupados por ala.
- Ocupação total por ala (`ocupacao_ala`).
- Alerta de sensores (se há inconsistência).
- Qualidade do ar.
- Velocidade atual da ventoinha.
- “Vai encher” / “vai aliviar” (previsão perto do tempo real).

**Corre em:** Raspberry Pi  
**Linguagem / stack possível:**
- Opção rápida: Node-RED (fluxos visuais + JS mínimo)
- Opção personalizada: Flask (Python) + HTML/CSS/JS

---

## 3. Comunicação e Mensagens

Os nós (Arduinos) comunicam com o Raspberry Pi usando MQTT (ou série, se quisermos começar simples).  
Formato típico de mensagem (JSON):

```json
{
  "ala": "A",
  "lugar": "A-03",
  "estado_lugar": "ocupado",
  "ocupacao_ala": 12,
  "qualidade_ar": 410,
  "ventoinha_percent": 60,
  "alerta_sensor": false,
  "timestamp": "2025-10-30T12:15:00Z"
}

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
No Raspberry Pi, em Python:
- Usa hora do dia e dia da semana (padrões de utilização).
- Usa ocupação média dos últimos 30–60 minutos.
- Usa o fluxo recente na ENTRADA (quantos carros entraram nos últimos 5–15 min).
- Com isso, calcula se a ala vai encher ou vai aliviar nos próximos minutos.

Essa previsão:
- aparece no dashboard (“vai encher” / “vai aliviar”)
- e é usada para controlar a ventoinha.

### 4.3 Ventilação antecipada (feed-forward)
Cada ala tem um sensor de ar (por exemplo MQ-7 para CO).

Lógica:
1. Se a previsão diz que a ala vai encher → o Raspberry Pi manda subir a velocidade base da ventoinha (por ex. 60%, 80%).
2. Se a previsão diz que vai acalmar → manda baixar para o mínimo (ex. 20%).
3. Segurança:
   - Se o ar já está mau → ventoinha 100%.
   - Se o ar está bom → fica no mínimo.

Isto é feed-forward: reagir ANTES do ar ficar mau.  
O Raspberry Pi envia a % desejada e o Arduino da ala gera o PWM físico.

---

## 5. Resumo das Linguagens e Onde Corre

| Parte                              | Corre onde          | Linguagem / Ferramentas                   |
|-----------------------------------|---------------------|-------------------------------------------|
| Nó do lugar (sensor + LED)        | Arduino UNO R4 WiFi | C/C++ (Arduino IDE / PlatformIO)          |
| Nó da ala (entrada/saída/ar/fan)  | Arduino UNO R4 WiFi | C/C++ (Arduino)                           |
| Gateway / previsão / lógica       | Raspberry Pi        | Python                                    |
| Comunicação MQTT                  | Raspberry Pi        | Mosquitto (broker MQTT) + libs C++/Python |
| Dashboard                         | Raspberry Pi        | Node-RED (JS simples) ou Flask (Python + HTML/JS) |
| Base de dados / histórico         | Raspberry Pi        | Python (SQLite ou InfluxDB)               |

---

## 6. O que vamos demonstrar

No final queremos conseguir mostrar:

1. Um lugar que deteta se está ocupado ou livre, acende LED e envia estado.
2. Uma ala que tem ENTRADA e SAÍDA físicas separadas e mantém `ocupacao_ala` com ++ e --.
3. O sistema a detetar diferença entre `ocupacao_ala` e a soma dos lugares ocupados, e gerar `alerta_sensor`.
4. O Raspberry Pi a prever que a ala vai encher e a mandar subir a ventoinha antes do ar ficar mau.
5. Um dashboard com tudo em tempo real (ocupação, previsão, ar, aviso de abrir nova ala).

Isto junta eletrónica, firmware, comunicação, previsão e interface — um projeto IoT completo.