
set1=[]
ss=[1,2,3,4,5,6,7,8,9]
for j in ss:
    for k in ss:
        for l in ss:
            test=(1000*j+100*k+10*k+l)
            re=test%9
            r2=(10*j+k)%9
            r3=(10*k+l)%9
            if re==0 and  r2==0 and r3==0 :
                set1.append(100*j+10*k+l)

print(set1)
set=set(set1)
print(set)
print(len(set1))
