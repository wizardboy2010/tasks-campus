const int motorpin1 = 4;                  //define digital output pin no.
const int motorpin2 = 3;                  //define digital output pin no.
int value = 250;

void setup () {
  
  pinMode(motorpin1,OUTPUT);        //set pin 3 as output
  pinMode(motorpin2,OUTPUT);        // set pin 4 as output
  Serial.begin(9600);
}

void loop () {

  digitalWrite(motorpin1,LOW);
  digitalWrite(motorpin2,value);

}
