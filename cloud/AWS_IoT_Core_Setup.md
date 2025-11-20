# Guia de Integração com AWS IoT Core

Este documento descreve os passos necessários para reproduzir a integração da tarefa 3.1 na AWS. Segue a ordem para evitar erros.

---

## 1. Pré-requisitos

- Conta AWS com permissões para IoT Core, IAM, DynamoDB/Timestream (ou outro serviço de storage) e API Gateway/Lambda.
- AWS CLI configurada **ou** acesso ao console web.
- Nome lógico para a ala (ex.: `alaA`).
- Certificados gerados pela AWS IoT (serão descarregados para o Pi).

---

## 2. IoT Core – criação do ambiente

1. **Criar um “Thing” (dispositivo)**
   - IoT Core → *Manage* → *Things* → *Create thing*.
   - Single thing → Nome: `alaA_gateway`.
   - Tipo: `Gateway`.

2. **Certificados e chaves**
   - Durante a criação, gera um novo certificado.
   - Faz download de:
     - `certificate.pem.crt`
     - `private.pem.key`
     - `AmazonRootCA1.pem` (CA pública)
   - Anexa uma “policy” ao certificado (ver passos seguintes).

3. **Política IoT (IAM)**
   - Cria uma policy com permissões mínimas. Exemplo:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Effect": "Allow",
           "Action": [
             "iot:Publish",
             "iot:Subscribe",
             "iot:Receive",
             "iot:Connect"
           ],
           "Resource": [
             "arn:aws:iot:REGION:ACCOUNT_ID:topic/aws/alaA/*",
             "arn:aws:iot:REGION:ACCOUNT_ID:topicfilter/aws/alaA/*",
             "arn:aws:iot:REGION:ACCOUNT_ID:client/alaA-gateway"
           ]
         }
       ]
     }
     ```
   - Substitui `REGION` e `ACCOUNT_ID`.
   - Anexa esta policy ao certificado gerado anteriormente.

4. **Tópicos MQTT**
   - Define a convenção usada pelo gateway:
     - `aws/alaA/telemetry` (publicação Pi → cloud)
     - `aws/alaA/commands` (comandos Cloud → Pi)
     - `aws/alaA/lugares` (estado detalhado dos lugares)
     - `aws/alaA/estado` (estado agregado da ala)
      - `aws/alaA/alertas` (alertas: ala cheia, etc.)
     - `aws/alaA/daily_forecast` (previsão diária em JSON)
   - Não é necessário criar formalmente os tópicos; basta garantir que as policies permitem o acesso.

5. **Testar ligação**
   - Em *MQTT test client*, subscreve `aws/alaA/#`.
   - Publica manualmente uma mensagem em `aws/alaA/commands` para garantir que chega ao tópico (verificados mais tarde no Pi).

---

## 3. Pipeline de ingestão na cloud

Escolhe uma base de dados gerida (exemplos abaixo). O objetivo é guardar as mensagens recebidas de `aws/alaA/telemetry`.

### Opção A – DynamoDB
1. Cria uma tabela `ParkingTelemetry` com chave primária `pk` (STRING) e `sk` (STRING) ou outro esquema conveniente.
2. Cria uma *AWS Lambda* (Python 3.11) que recebe eventos do IoT Core:
   ```python
   import json
   import os
   import boto3
   from datetime import datetime

   dynamo = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])

   def handler(event, context):
       # event["payload"] vem em base64 → decodifica
       payload = json.loads(event["payload"].decode("utf-8"))
       timestamp = payload.get("timestamp") or datetime.utcnow().isoformat()
       dynamo.put_item(
           Item={
               "pk": f"ala#{payload['summary']['ala']}",
               "sk": timestamp,
               "payload": payload,
           }
       )
       return {"statusCode": 200}
   ```
3. Configura uma *IoT Rule*:
   - SQL: `SELECT * FROM 'aws/alaA/telemetry'`
   - Ação: enviar para a Lambda (`lambda:InvokeFunction`).

### Opção B – Timestream
1. Cria uma base Timestream, tabela `Parking`.
2. Lambda semelhante, mas a escrever via `boto3.client("timestream-write")`.
3. Cria uma *Query* ou *Scheduled Query* para agregações.

> Documenta a escolha (DynamoDB vs Timestream) no relatório; a lógica acima cobre a tarefa “Implementar função/broker para gravar dados em DB gerida”.

---

## 4. API/Endpoint para comandos (cloud → Pi)

1. **API Gateway HTTP API**
   - Cria uma API HTTP.
   - Rota `POST /commands`.
2. **Lambda de comandos**
   ```python
   import json
   import boto3
   import os

   iot = boto3.client("iot-data", endpoint_url=f"https://{os.environ['IOT_ENDPOINT']}")

   def handler(event, context):
       body = json.loads(event.get("body") or "{}")
       # Espera algo como {"ala": "A", "percent": 80}
       topic = os.environ["COMMAND_TOPIC"]
       iot.publish(topic=topic, qos=1, payload=json.dumps(body))
       return {"statusCode": 202, "body": json.dumps({"status": "accepted"})}
   ```
3. **Integração**
   - API Gateway → Lambda (proxy).
   - Definir API Key se quiseres restringir (combina com `rest.api_key` no `config.yaml`).
4. **Resposta Pi**
   - O gateway já subscreve `aws/alaA/commands`; qualquer `POST` na API publica nesse tópico, fechando o ciclo Cloud → Pi.

---

## 5. Retenção e limpeza

- No Pi, a configuração `cloud.retention.retention_days` controla quando os registos locais são purgados.
- Na AWS, define políticas:
  - **DynamoDB**: ativa TTL ou tarefas de limpeza (somente se usares atributos TTL).
  - **Timestream**: configura “Magnetic Store retention” e “Memory Store retention”.
  - **S3/Data Lake** (se exportares): cria regras de lifecycle.

Regista no relatório a política adotada (ex.: 30 dias no Pi, 180 dias na cloud).

---

## 6. Configuração no Raspberry Pi

1. Copia os ficheiros de certificado para `/home/pi/iot_gateway/certs/`.
2. Garante que `config.yaml` usa os caminhos corretos:
   ```yaml
   cloud:
     provider: aws_iot_core
     mqtt:
       enabled: true
       endpoint: a1b2c3d4e5f6-ats.iot.eu-west-1.amazonaws.com
       client_id: alaA-gateway
       ca_cert: /home/pi/iot_gateway/certs/AmazonRootCA1.pem
       certfile: /home/pi/iot_gateway/certs/device.pem.crt
       keyfile: /home/pi/iot_gateway/certs/private.pem.key
       topics:
         cloud_out: aws/alaA/telemetry
         cloud_in: aws/alaA/commands
         lugar: aws/alaA/lugares
         ala: aws/alaA/estado
   ```
3. Se usares API Gateway (REST), ativa:
   ```yaml
   rest:
     enabled: true
     base_url: https://<api-id>.execute-api.<region>.amazonaws.com/prod/telemetry
     api_key: <se necessário>
   ```
4. Reinicia o gateway (`systemctl restart iot-gateway` ou `python gateway_pi/main.py ...`) e verifica logs:
   - “Cliente MQTT cloud ligado...”
   - “Dados sincronizados com a cloud...”
   - Recebe comandos publicados pela API/gateway AWS.

---

## 7. Checklist final

- [ ] Thing + Certificates + Policy configurados no IoT Core.
- [ ] Lambda/IoT Rule a escrever em DynamoDB ou Timestream.
- [ ] API Gateway + Lambda para comandos (opcional mas recomendado).
- [ ] Certificados copiados para o Pi e configurados no `config.yaml`.
- [ ] MQTT test client confirma tráfego bidirecional (`telemetry` / `commands`).
- [ ] Logs do gateway indicam envio para MQTT cloud e/ou REST.
- [ ] Tabela/bucket na cloud está a receber dados.
- [ ] Estratégia de retenção documentada (Pi + cloud).

Cumprindo estes pontos, a tarefa 3.1 fica operacional: Pi → Cloud (telemetria), Cloud → Pi (comandos), dados persistidos numa base gerida e política de retenção assegurada.


