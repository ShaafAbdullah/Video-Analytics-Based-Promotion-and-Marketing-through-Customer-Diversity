import mysql.connector
def gender(num):

  #mySQLconnection = mysql.connector.connect(host="localhost", user="root",passwd="",database="fyp")
  mySQLconnection = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database="fyp"
)
  mycursor = mySQLconnection.cursor()
  #number=[903,455]
  inr=1;
  for i in range (2):
   print(i,num[i],inr)

   sql = "UPDATE  count_gender SET number = %s WHERE id = %s"
   val = (num[i], inr)
   mycursor.execute(sql, val)
   mySQLconnection.commit()
   inr=inr+1
   print(mycursor.rowcount, "record(s) affected")

def age(num):

  #mySQLconnection = mysql.connector.connect(host="localhost", user="root",passwd="",database="fyp")
  mySQLconnection = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="",
      database="fyp"
)
  mycursor = mySQLconnection.cursor()
  #number=[903,455]
  inr=1;
  for i in range (3):
   print(i,num[i],inr)

   sql = "UPDATE  count_age_group SET number = %s WHERE id = %s"
   val = (num[i], inr)
   mycursor.execute(sql, val)
   mySQLconnection.commit()
   inr=inr+1
   print(mycursor.rowcount, "record(s) affected")




def peopele(num):

  #mySQLconnection = mysql.connector.connect(host="localhost", user="root",passwd="",database="fyp")
  mySQLconnection = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="",
      database="fyp"
)
  mycursor = mySQLconnection.cursor()
  #number=[903,455]
  inr=1;
  for i in range (3):
   print(i,num[i],inr)

   sql = "UPDATE  pp SET number = %s WHERE id = %s"
   val = (num[i], inr)
   mycursor.execute(sql, val)
   mySQLconnection.commit()
   inr=inr+1
   print(mycursor.rowcount, "record(s) affected")


