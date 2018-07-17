int ledPinRed = 2;
int Button = A0;
int val = 0;
void setup() {
  // put your setup code here, to run once:
Serial.begin(9600);
pinMode(ledPinRed, OUTPUT);
pinMode(Button, INPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  
int val = analogRead(Button);

//Serial.println(val);
//delay(1000);
if(val/1000 == HIGH)
{
  digitalWrite(ledPinRed, HIGH);
}
else
{
  digitalWrite(ledPinRed, LOW);
}
}
