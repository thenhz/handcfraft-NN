import numpy as np
from nhzworks.LSTM_Network import RecurrentNeuralNetwork

def LoadText():
    # open text and return input and output data (series of words)
    text = "Mi guardo allo specchio, arrabbiata e delusa. Al diavolo i miei capelli, che non vogliono saperne di stare a posto, e al diavolo Katherine Kavanagh, che si è ammalata e mi sottopone a questa prova. Dovrei studiare per gli ultimi esami, che saranno la settimana prossima, e invece eccomi qui a cercare di domare questa chioma ribelle. “Non devo più andare a letto con i capelli bagnati. Non devo più andare a letto con i capelli bagnati.” Recitando più volte questo mantra tento, di nuovo, di addomesticarli con la spazzola. Contemplo esasperata la diafana ragazza castana con gli occhi azzurri, troppo grandi per il suo viso, che mi fissa dallo specchio, e depongo le armi. La mia unica possibilità è legarli in una coda e sperare di avere un aspetto almeno presentabile. Kate è la mia coinquilina, e fra tutti i giorni possibili ha scelto proprio questo per farsi venire l’influenza. Così, non può fare l’intervista, programmata per il giornale studentesco, a un pezzo grosso dell’industria che io non ho mai sentito nominare, e mi sono dovuta offrire di andarci al posto suo. Ho gli esami da preparare, una tesina da finire, e nel pomeriggio dovrei presentarmi al lavoro, ma no… oggi mi tocca guidare per più di duecento chilometri fino a Seattle per incontrare il misterioso amministratore delegato della Grey Enterprises Holdings Inc. Il tempo di questo eccezionale imprenditore, nonché importante sponsor della nostra università, è straordinariamente prezioso – molto più del mio – ma ciò non gli ha impedito di concedere a Kate un’intervista. Un vero scoop, mi dice lei. Al diavolo la mia amica e le sue attività extracurricolari. Kate è raggomitolata sul divano del soggiorno. «Ana, mi dispiace. Mi ci sono voluti nove mesi per ottenere questa intervista. Ce ne vorrebbero altri sei per spostare l’appuntamento, e a quel punto saremo entrambe laureate. Come direttore del giornale, non posso giocarmi questa chance. Ti prego» mi implora con la voce rauca per il mal di gola. Ma come fa? Anche da malata è uno schianto, con i capelli ramati in perfetto ordine e gli occhi verdi splendenti, anche se adesso sono cerchiati di rosso e lacrimano. Ignoro un inopportuno moto di compassione. «Certo che ci andrò, Kate. Ora è meglio che tu torni a letto. Vuoi un po’ di NyQuil o di Tylenol?» «NyQuil, grazie. Qui ci sono le domande e il registratore. Basta che premi questo pulsante. Prendi appunti, poi trascriverò tutto io.» «Non so niente di quel tizio» mormoro, cercando invano di reprimere il panico. «Basta che segui l’ordine delle domande. Adesso vai. Il viaggio è lungo. Non vorrei che arrivassi in ritardo.» «Va bene, vado. Tu torna a letto. Ti ho preparato una zuppa da scaldare.» Le lancio uno sguardo pieno d’affetto. “Lo faccio solo perché sei tu, Kate.” «D’accordo. In bocca al lupo. E grazie, Ana… Come al solito, mi salvi la vita.» Mentre prendo lo zainetto, le rivolgo un sorriso tirato, esco e mi dirigo verso l’auto. Non posso credere di essermi lasciata convincere a fare questa pazzia. D’altra parte Kate convincerebbe chiunque a fare qualsiasi cosa. Diventerà una grande giornalista"
    text = list(text)
    outputSize = len(text)
    text = list(set(text))
    uniqueWords, dataSize = len(text), len(text)
    returnData = np.zeros((uniqueWords, dataSize))
    for i in range(0, dataSize):
        returnData[i][i] = 1
    returnData = np.append(returnData, np.atleast_2d(text), axis=0)
    output = np.zeros((uniqueWords, outputSize))
    for i in range(0, outputSize):
        index = np.where(np.asarray(text) == text[i])
        output[:,i] = returnData[0:-1,index[0]].astype(float).ravel()
    return returnData, uniqueWords, output, outputSize, text
#write the predicted output (series of words) to disk
def ExportText(output, data):
    finalOutput = np.zeros_like(output)
    prob = np.zeros_like(output[0])
    outputText = ""
    print(len(data))
    print(output.shape[0])
    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            prob[j] = output[i][j] / np.sum(output[i])
        outputText += np.random.choice(data, p=prob)
    print(outputText)
    return




#Begin program
print("Beginning")
iterations = 100
learningRate = 0.001
# parameters
# n_steps = seq_len-1
# n_inputs = 4
# n_neurons = 200
# n_outputs = 4
# n_layers = 2
# learning_rate = 0.001
# batch_size = 50
# n_epochs = 100
# train_set_size = x_train.shape[0]
# test_set_size = x_test.shape[0]
#load input output data (words)
returnData, numCategories, expectedOutput, outputSize, data = LoadText()
print("Done Reading")
#init our RNN using our hyperparams and dataset
RNN = RecurrentNeuralNetwork(numCategories, numCategories, outputSize, expectedOutput, learningRate)
#training time!
for i in range(1, iterations):
    #compute predicted next word
    RNN.forwardProp()
    #update all our weights using our error
    error = RNN.backProp()
    #once our error/loss is small enough
    print("Error on iteration ", i, ": ", error)
    if error > -100 and error < 100 or i % 100 == 0:
        #we can finally define a seed word
        seed = np.zeros_like(RNN.x)
        maxI = np.argmax(np.random.random(RNN.x.shape))
        seed[maxI] = 1
        RNN.x = seed
        #and predict some new text!
        output = RNN.sample()
        print(output)
        #write it all to disk
        ExportText(output, data)
        print("Done Writing")
print("Complete")
