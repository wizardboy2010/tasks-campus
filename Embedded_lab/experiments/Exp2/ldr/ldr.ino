const int Pin = 7;
int value = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  //pinMode(Pin, INPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  value = digitalRead(Pin);
  Serial.println(value);
  delay(100);

}
