import processing.serial.*;

Serial myPort;  // Create object from Serial class
String val;     // Data received from the serial port
int flag = 0;

void setup()
{
  // I know that the first port in the serial list on my mac
  // is Serial.list()[0].
  // On Windows machines, this generally opens COM1.
  // Open whatever port is the one you're using.
  String portName = Serial.list()[0]; //change the 0 to a 1 or 2 etc. to match your port
  myPort = new Serial(this, portName, 9600);
}


void draw()
{
  if ( myPort.available() > 0) 
  {  // If data is available,
  val = myPort.readStringUntil('\n');         // read it and store it in val
  }
  //float temp = new Float(val).floatValue();
  //float temp = Float.valueOf(val);
  //float temp = Float.parseFloat(val);
 //println(val.length()); 
  println(val); //print it out in the console
  
  if (val == "Motion detected!")
    {
      flag = 1;
    }
  if (val == "Motion stopped!")
    {
      flag = 0;
    }
    if (float(val)>1000)
      {
        color c = color(255, 204, 0);  // Define color 'c'
        fill(c);  // Use color variable 'c' as fill color
        noStroke();  // Don't draw a stroke around shapes
        rect(30, 20, 55, 55);  // Draw rectangle
      }
      else {
        color c = color(0, 204, 255);  // Define color 'c'
        fill(c);  // Use color variable 'c' as fill color
        noStroke();  // Don't draw a stroke around shapes
        rect(30, 20, 55, 55);  // Draw rectangle
      }
    
}
