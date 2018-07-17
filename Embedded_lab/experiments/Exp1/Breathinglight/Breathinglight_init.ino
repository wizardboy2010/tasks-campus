int ledPin = 5;
int bright = 0;
int increment = 10;
void setup() {
  // put your setup code here, to run once:
pinMode(ledPin, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:

for( int i = 0; i < 255; i + increment)
{
  bright = bright + increment;
  analogWrite(ledPin, bright);
  delay(100);
 } 
  
}
