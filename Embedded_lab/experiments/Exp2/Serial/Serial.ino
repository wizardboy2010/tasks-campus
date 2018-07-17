int led = 13;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(led, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  int light = Serial.read();
  while(light != 0)
  {
    analogWrite(led, light);
  }
