// Initialising the pins and variables
int ledPinRed = 2;
int Button = 7;
int flag  = 0;
int val = 0;

void setup() {
  // setting up pins configuration 
pinMode(ledPinRed, OUTPUT);
pinMode(Button, INPUT);
}

void loop() {
  // Main code is here

// reading the statuts of the button
int val = digitalRead(Button);

// flag of the led is updated when we press the keypad...0 or 1
if(val == HIGH)
{
flag = 1 - flag;
}

// If flag is 1 bulb is on until flag turned to 0
if(flag == 1)
{
  digitalWrite(ledPinRed, HIGH);
}
else
{
  digitalWrite(ledPinRed, LOW);
}
}
