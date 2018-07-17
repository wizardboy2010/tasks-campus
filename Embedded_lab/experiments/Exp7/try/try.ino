int pin = A0;

void setup() {
  // put your setup code here, to run once:
  DDRD = DDRD | B11111111;
  pinMode(pin,INPUT);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  PORTD = PORTD | B00000100;
  Serial.println(PIND, BIN);
  //Serial.println(PIND, BIN);
  //Serial.println(analogRead(pin));
  delay(100);
}
