enum PayloadState {
  PAYLOAD_CLOSED = 0,
  PAYLOAD_OPEN = 1,
};

static constexpr unsigned long SERIAL_BAUD = 9600;
static constexpr size_t MAX_COMMAND_LENGTH = 64;

PayloadState currentState = PAYLOAD_OPEN;
String commandBuffer;

void applyStateToLed() {
  digitalWrite(LED_BUILTIN, currentState == PAYLOAD_OPEN ? HIGH : LOW);
}

void sendState() {
  if (currentState == PAYLOAD_OPEN) {
    Serial.println("STATE:OPEN");
    return;
  }

  Serial.println("STATE:CLOSED");
}

void handleCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command.length() == 0) {
    return;
  }

  if (command == "OPEN") {
    currentState = PAYLOAD_OPEN;
    applyStateToLed();
    sendState();
    return;
  }

  if (command == "CLOSE") {
    currentState = PAYLOAD_CLOSED;
    applyStateToLed();
    sendState();
    return;
  }

  if (command == "STATE?") {
    sendState();
    return;
  }

  Serial.println("ERROR:UNKNOWN_COMMAND");
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  applyStateToLed();

  Serial.begin(SERIAL_BAUD);
  commandBuffer.reserve(MAX_COMMAND_LENGTH);

  unsigned long startMs = millis();
  while (!Serial && (millis() - startMs) < 2000UL) {
    delay(10);
  }
}

void loop() {
  while (Serial.available() > 0) {
    char incoming = static_cast<char>(Serial.read());

    if (incoming == '\r') {
      continue;
    }

    if (incoming == '\n') {
      handleCommand(commandBuffer);
      commandBuffer = "";
      continue;
    }

    if (commandBuffer.length() >= MAX_COMMAND_LENGTH) {
      commandBuffer = "";
      Serial.println("ERROR:COMMAND_TOO_LONG");
      continue;
    }

    commandBuffer += incoming;
  }
}
