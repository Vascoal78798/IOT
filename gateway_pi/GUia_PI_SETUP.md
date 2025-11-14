# Guia Completo para pôr o Gateway a funcionar no Raspberry Pi

Este guia leva-te, passo a passo, desde o hardware até ao software, para teres a tarefa 3 (Gateway / Raspberry Pi) totalmente funcional. Lê de seguida e não saltes passos.

---

## 1. O que precisas

- Raspberry Pi (recomendado: 3B+ ou superior) com fonte oficial 5 V/3 A e cartão microSD com Raspberry Pi OS atualizado.
- 2 × Arduino UNO R4 WiFi (1 = nó do lugar, 1 = nó da ala).
- Sensores ultrassónicos (1 para o lugar, 2 para a ala entrada/saída).
- Sensor MQ-7 (qualidade do ar) + aquecedor e alimentação conforme datasheet.
- Ventoinha ou relé + alimentação externa (com GND comum ao Arduino).
- LEDs verde/vermelho para o lugar.
- Cabos USB (Arduino ⇄ Pi) e jumpers.
- PC com Arduino IDE para gravares os sketches.
- Acesso à internet local (para MQTT e atualização de pacotes).

---

## 2. Flash dos Arduinos (no PC)

1. Abre o Arduino IDE e instala o core da placa UNO R4 WiFi (Board Manager).
2. Liga o **nó do lugar** ao PC por USB.
   - Abre `node_lugar/node_lugar.ino`.
   - Confirma que a porta série está certa (Menu: Ferramentas → Porta).
   - Carrega o sketch (`Ctrl+U`).
3. Desliga o Arduino e volta a ligar o **nó da ala**.
   - Abre `node_ala/node_ala.ino`.
   - Volta a carregar (`Ctrl+U`).
4. Teste rápido no IDE: abre o Monitor Série a 115 200 bps para verificar se cada sketch escreve mensagens de arranque.

> **Dica:** aponta num papel qual Arduino será usado para o lugar e qual para a ala para não trocarem quando fores para o Pi.

---

## 3. Ligações de hardware

### Nó do lugar (Arduino UNO R4 WiFi)
- Sensor ultrassónico: `TRIGGER_PIN` = 2, `ECHO_PIN` = 3 (ver sketch).
- LED verde: pino 9.
- LED vermelho: pino 10.
- Alimentação 5 V e GND a partir do Arduino.

### Nó da ala (Arduino UNO R4 WiFi)
- Sensor ultrassónico entrada → pino A0.
- Sensor ultrassónico saída → pino A1.
- MQ-7 → pino A2 (liga também o aquecedor conforme datasheet).
- Ventoinha/relé → pino 9 (PWM, tens de ter alimentação externa se a ventoinha puxar mais corrente).
- Alimentação 5 V e GND do Arduino; une o GND da ventoinha ao GND do Arduino.

### Raspberry Pi
- Alimentado com a fonte 5 V/3 A.
- Ligação de rede (Ethernet ou Wi-Fi).
- Liga os **dois Arduinos ao Pi por USB** (idealmente fios curtos e directos).

---

## 4. Preparar o Raspberry Pi

### 4.1. Atualizar o sistema e instalar pacotes base
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install git python3-venv python3-pip mosquitto mosquitto-clients sqlite3 -y
```

### 4.2. Clonar ou atualizar o repositório
```bash
cd /home/pi
git clone https://github.com/SEU_UTILIZADOR/IOT.git  # se ainda não tiveres
# ou, se já existe:
cd IOT
git pull
```

> Ajusta o URL se estiveres a usar um repositório privado (HTTPS ou SSH).

### 4.3. Criar ambiente virtual Python
```bash
cd /home/pi/IOT
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r gateway_pi/requirements.txt
```

> **TinyML**: instala `tflite-runtime` manualmente (varia conforme arquitectura). Exemplo para Pi 4 ARMv7:
> ```bash
> pip install https://github.com/google-coral/edgetpu/releases/download/release-frogfish/tflite_runtime-2.12.0-cp39-cp39-linux_armv7l.whl
> ```
> Se não tiveres modelo TinyML ainda, podes fazer esta parte mais tarde.

---

## 5. Identificar portas série dos Arduinos no Pi

1. Liga **apenas o nó do lugar** ao Pi por USB:
   ```bash
   ls /dev/ttyACM*
   # Exemplo: /dev/ttyACM0
   ```
   Aponta o nome da porta.
2. Liga também o nó da ala e repete:
   ```bash
   ls /dev/ttyACM*
   # Exemplo: /dev/ttyACM0 /dev/ttyACM1
   ```
3. Para saber qual é qual, usa `dmesg --follow` ou o comando abaixo enquanto desconectas/ligas cada um:
   ```bash
   watch -n1 ls /dev/ttyACM*
   ```
4. Anota resultados definitivos (ex.: `/dev/ttyACM0 = lugar`, `/dev/ttyACM1 = ala`). Se em cada arranque alterarem, toma atenção e ajusta o YAML conforme necessário.

---

## 6. Configurar o gateway

### 6.1. Criar ficheiro de configuração
```bash
cp gateway_pi/config_example.yaml gateway_pi/config.yaml
nano gateway_pi/config.yaml
```

### 6.2. Editar `config.yaml`
- `sqlite_db_path`: onde queres guardar a base de dados (por omissão `/home/pi/iot_gateway/iot.db` – podes deixar).
- Secção `serial_devices`:
  ```yaml
  serial_devices:
    - path: /dev/ttyACM0
      tipo: lugar
      id_lugar: A-01
    - path: /dev/ttyACM1
      tipo: ala
      id_ala: A
  ```
  Ajusta os `path` conforme o passo 5 e, se tiveres mais lugares/alas, adiciona entradas.
- Secção `alas`:
  - `capacidade_maxima` e `soma_lugares_inicial` conforme a tua ala real.
  - Em `safety`, define:
    - `default_percent`: percentagem de partida da ventoinha.
    - `occupancy_rules`: thresholds de ocupação.
    - `qualidade_ar_threshold` e `qualidade_ar_percent` para quando o MQ-7 detecta ar mau.
    - `mismatch_delta` e `mismatch_duration_seconds` para disparar `alerta_sensor` se a contagem não bater certo com os lugares.
    - `ack_timeout_seconds`: quanto tempo esperar por ACK do Arduino após enviar comando.
- Secção `mqtt`:
  - Se tiveres broker local no Pi (mosquitto), podes deixar `127.0.0.1:1883`.
  - Ajusta os tópicos se necessário (ex.: `cloud_out`, `cloud_in`).
- Secção `tinyml` (opcional):
  - Aponta `dataset_path`, `model_path` e `metrics_path` para locais reais (por ex. `/home/pi/iot_gateway/models/alaA.tflite`).
  - Se ainda não tens modelo, podes pôr `enabled: false` temporariamente.
- Secção `cloud`:
  - Para AWS IoT Core segue o guia `cloud/AWS_IoT_Core_Setup.md` (criação do Thing, tópicos, certificados, API).
  - Introduz os caminhos dos certificados e os tópicos que definiste na plataforma escolhida.

Grava (`Ctrl+O`, Enter) e sai (`Ctrl+X`).

---

## 7. Preparar dataset e modelo (opcional)

Se tens dados históricos e queres treinar no Pi (requer TensorFlow completo – pode ser pesado):
```bash
source /home/pi/IOT/.venv/bin/activate
python gateway_pi/tinyml_pipeline.py export --config gateway_pi/config.yaml
python gateway_pi/tinyml_pipeline.py train --config gateway_pi/config.yaml --epochs 120
```
Isto cria/actualiza o dataset e o `.tflite`. Se o Pi não aguentar, treina num PC e copia o ficheiro `.tflite` para o caminho indicado no YAML (via `scp`, pendrive, etc.).

---

## 8. Execução do gateway

### 8.1. Iniciar manualmente
```bash
source /home/pi/IOT/.venv/bin/activate
python gateway_pi/main.py --config gateway_pi/config.yaml --log-level INFO
```
- Mantém esta janela aberta para veres os logs. Deves ver mensagens como “Gateway iniciado. A aguardar mensagens…”.
- Quando um Arduino enviar dados, aparecem logs tipo `[Serial]` ou `[Controlo:...]`.

### 8.2. (Opcional) Iniciar como serviço systemd
Cria `/etc/systemd/system/iot-gateway.service` com o conteúdo:
```ini
[Unit]
Description=IoT Gateway
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/IOT
ExecStart=/home/pi/IOT/.venv/bin/python gateway_pi/main.py --config /home/pi/IOT/gateway_pi/config.yaml --log-level INFO
Restart=on-failure
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```
Depois:
```bash
sudo systemctl daemon-reload
sudo systemctl enable iot-gateway
sudo systemctl start iot-gateway
sudo journalctl -fu iot-gateway
```
Assim, o gateway arranca automaticamente ao ligar o Pi.

---

## 9. Testes essenciais

1. **Conectividade série**: desconecta e volta a ligar cada Arduino → verifica no log que o gateway identifica a porta e retoma a leitura.
2. **Lugar**: mete/retira objeto → LED muda, log `Lugar ... → ocupado/livre`, entrada em `place_events` (consulta com `sqlite3`).
3. **Ala**: passa objeto pela entrada/saída → log `entrada/saida`, contador actualiza.
4. **MQ-7**: aproxima uma fonte (ou simula com alteração temporária no código) → check se `qualidade_ar_tensao` sobe e a ventoinha chega ao valor de segurança.
5. **Mismatch**: altera manualmente `soma_lugares` (comando cloud ou firmware) para diferir de `ocupacao_ala` e aguarda `mismatch_duration` segundos → deve activar `alerta_sensor` e ventoinha 100 %.
6. **ACK ventoinha**: observa se após cada comando `percent` o nó responde com `{"evento":"ventoinha","percent":...}`; se desligares o Arduino, surge log “Sem ACK...” após o timeout.
7. **Overrides cloud**: (se quiseres) publica via MQTT `{"ala":"A","percent":80}` no tópico `cloud_in`. A ventoinha muda, mas depois do TTL volta a ser controlada pelas regras.
8. **TinyML**: quando tiveres modelo, verifica log `[TinyML] Ala ...` e confirma que `tinyml_predictions` regista as previsões.

---

## 10. Resumo rápido (checklist)

- [ ] Arduinos com firmware certo (lugar/ala) e cablagem ok.
- [ ] Raspberry Pi atualizado, com git, mosquitto, python-venv, etc.
- [ ] Repositório clonado/actualizado em `/home/pi/IOT`.
- [ ] Ambiente virtual criado e dependências instaladas.
- [ ] `config.yaml` preenchido com portas série correctas, safety rules e MQTT.
- [ ] (Opcional) TinyML `.tflite` copiado e caminhos configurados.
- [ ] Gateway executa sem erros e dá logs coerentes.
- [ ] Testes básicos (LEDs, contagem, MQ-7, alertas, overrides) verificados.
- [ ] Base de dados `iot_gateway.db` a registar eventos.
- [ ] (Opcional) Serviço systemd configurado para arranque automático.

Cumprindo esta lista ficas com a tarefa 3 pronta para demonstrar: 
- nó do lugar reporta ocupação com histerese; 
- nó da ala conta veículos, valida qualidade de ar, recebe comandos; 
- Pi centraliza dados, aplica regras de segurança, aceita overrides/preditivos e sincroniza via MQTT/cloud.

Boa configuração! Se precisares de adaptar para mais alas ou integrar outro protocolo (HTTP/REST), usa este setup como base.


