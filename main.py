from util import play, test_model

p = play()
m = test_model()

if(p > m):
    print("Model wins!")
elif(m == p):
    print("Draw!")
else:
    print("You win!")