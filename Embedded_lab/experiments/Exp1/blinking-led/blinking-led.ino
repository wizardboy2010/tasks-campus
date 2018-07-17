// Initialising Pin
int Redled= 8 ;
int Orangeled = 2;
void setup() {
  // Configuring Pin mode
  pinMode(Redled,OUTPUT);
  pinMode(Orangeled, OUTPUT);
}

void loop() {
  // Turning on Red led and Orange led off for a sec
  digitalWrite(Redled,HIGH);
  digitalWrite(Orangeled,LOW);
  delay(1000);
  
  // Turning on Orange led and red led off for other sec
  digitalWrite(Redled,LOW);
  digitalWrite(Orangeled,HIGH);
  delay(1000);
}
