import pymysql
def iud(qry,val):
    con=pymysql.connect(host="localhost",user="root",password="",port=3306,db="strsprdcn")
    cmd=con.cursor()
    cmd.execute(qry,val)
    id=cmd.lastrowid
    con.commit()
    con.close()
    return id

def selectall(qry,val):
    con=pymysql.connect(host="localhost",user ="root",password="",port=3306,db="strsprdcn")
    cmd=con.cursor()
    cmd.execute(qry,val)
    res=cmd.fetchall()
    return res

def selectall2(qry):
    con=pymysql.connect(host="localhost",user="root",password="",port=3306,db="strsprdcn")
    cmd=con.cursor()
    cmd.execute(qry)
    res=cmd.fetchall()
    return res

def selectone1(qry):
    con = pymysql.connect(host="localhost",user="root",password="",port=3306,db="strsprdcn")
    cmd = con.cursor()
    cmd.execute(qry)
    res=cmd.fetchone()
    return res

def selectone(qry,val):
    con = pymysql.connect(host="localhost",user="root",password="",port=3306,db="strsprdcn")
    cmd = con.cursor()
    cmd.execute(qry,val)
    res=cmd.fetchone()
    return res
