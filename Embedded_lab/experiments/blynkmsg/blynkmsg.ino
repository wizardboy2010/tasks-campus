#define BLYNK_PRINT Serial


#include <SoftwareSerial.h>
SoftwareSerial SwSerial(0, 1); // RX, TX
    
#include <BlynkSimpleSerialBLE.h>
#include <SoftwareSerial.h>

// You should get Auth Token in the Blynk App.
// Go to the Project Settings (nut icon).
char auth[] = "51f1ca692cc043c2bfac0c2e290829df";

SoftwareSerial SerialBLE(0, 1); // RX, TX

void notifyOnButtonPress()
{
  // Invert state, since button is "Active LOW"
  int isButtonPressed = !digitalRead(2);
  if (isButtonPressed) {
    Serial.println("Button is pressed.");

    // Note:
    //   We allow 1 notification per 15 seconds for now.
    Blynk.notify("Yaaay... button is pressed!");
  }
}

void setup()
{
  // Debug console
  Serial.begin(9600);

  Blynk.begin(SerialBLE, auth);

  // Setup notification button on pin 2
  pinMode(2, INPUT_PULLUP);
  // Attach pin 2 interrupt to our handler
  attachInterrupt(digitalPinToInterrupt(2), notifyOnButtonPress, CHANGE);
}

void loop()
{
  Blynk.run();
}
