const int motorpin1 = 3;                  //define digital output pin no.
const int motorpin2 = A0;                  //define digital output pin no.
int value;

void setup () {
  
  pinMode(motorpin1,OUTPUT);        //set pin 3 as output
  pinMode(motorpin2,OUTPUT);        // set pin 4 as output
  Serial.begin(9600);
}

void loop () {

     value = Serial.parseInt();
     if (value !=0){
      digitalWrite(motorpin1,LOW);
     analogWrite(motorpin2,value);
     Serial.println(value);
     delay(1000);
     }
}
