// Initialising Pins and Variables
int led = 7;         
int brightness = 0;   
int fadeAmount = 5;  

void setup() {
  // Configuring Pins
  pinMode(led, OUTPUT);
}


void loop() {

  // Setting brightness of the led
  analogWrite(led, brightness);

  // Changing the brightness
  brightness = brightness + fadeAmount;

  // If the maximum brightness is reached slowly decrease the brightness
  if (brightness <= 0 || brightness >= 255) {
    fadeAmount = -fadeAmount;
  }

  // changing brightness with delay of 30ns
  delay(30);
}
