// Initialising Pins and Variables
int led = 7;         

void setup() {
  // Configuring Pins
  pinMode(led, OUTPUT);
}


void loop() {

  // Setting brightness of the led
  int brightness = 100;

  // Writing the brightness to led
  analogWrite(led, brightness);
  
}
