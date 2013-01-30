/**
 * Name: Steve Siska
 * login: siska
 * CS540-2
 * November 16, 2012
 * HW 3.4
 */

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */
import java.util.*;
public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {
	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */
	//Map to count occurences of a word per label
	private TreeMap<String, ArrayList<Double>> wordMap = new TreeMap<String, ArrayList<Double>>();
	//Size of vocabulary field to be used later, as stated in HW
	private int vocSize;
	private double numHam = 0; //number of HAM messages
	private double numSpam = 0; //number of SPAM messages
	private double hamTokens = 0; //number of HAM tokens
	private double spamTokens = 0; //number of SPAM tokens
	@Override
	/**
	 * This method should train your classifier with the training data provided. 
	 * The integer argument v is the size of the total vocabulary in your model. 
	 * 
	 * @param trainingData - an array of Instances (messages) to train the classifier
	 * @param v - the vocabulary size of the data
	 */
	public void train(Instance[] trainingData, int v) {
		vocSize = v;
		//put each word in the map, and count how many times it comes up in
		//SPAM messages and HAM messages
		for(int i = 0; i < trainingData.length; i++){
			for(int j = 0; j < trainingData[i].words.length; j++){
				//first check if the word is already in the map
				if(!wordMap.containsKey(trainingData[i].words[j])){
					if(trainingData[i].label == Label.HAM){
						wordMap.put(trainingData[i].words[j], new ArrayList<Double>());
						for(int n = 0; n < 4; n++){
							wordMap.get(trainingData[i].words[j]).add(0.0);
						}
						//since the map only accepts value changes of type ArrayList,
						//we need to store the List in a tmp variable to change it
						ArrayList<Double> tmp = wordMap.get(trainingData[i].words[j]);
						tmp.set(0, tmp.get(0)+1); //increment count of HAM
						wordMap.put(trainingData[i].words[j], tmp);
					}
					else{
						wordMap.put(trainingData[i].words[j], new ArrayList<Double>());
						for(int n = 0; n < 4; n++){
							wordMap.get(trainingData[i].words[j]).add(0.0);
						}
						ArrayList<Double> tmp = wordMap.get(trainingData[i].words[j]);
						tmp.set(1, tmp.get(1)+1); //increment count of SPAM
						wordMap.put(trainingData[i].words[j], tmp);
					}
				}
				else{
					if(trainingData[i].label == Label.HAM){
						ArrayList<Double> tmp = wordMap.get(trainingData[i].words[j]);
						tmp.set(0, tmp.get(0)+1); //increment count of HAM
						wordMap.put(trainingData[i].words[j], tmp);
					}
					else{
						ArrayList<Double> tmp = wordMap.get(trainingData[i].words[j]);
						tmp.set(1, tmp.get(1)+1); //increment count of SPAM
						wordMap.put(trainingData[i].words[j], tmp);
					}
				}
			}	
		}
		//now count the total occurences of SPAM and HAM messages in the data
		for(Instance i : trainingData){
			if(i.label == Label.SPAM){
				numSpam++;
			}
			else{
				numHam++;
			}
		}
		//set the probabilities in teh 3rd and 4th spots of each ArrayList
		for(Instance i : trainingData){
			for(int s = 0; s < i.words.length; s++){
				ArrayList<Double> tmp = wordMap.get(i.words[s]);
				double prob = p_w_given_l(i.words[s], i.label);
				int loc = i.label==Label.SPAM ? 3: 2;
				tmp.set(loc, prob);
				wordMap.put(i.words[s], tmp);
				int change = 0; //var to get which type we've just seen
				if(loc == 3) spamTokens++;
				else hamTokens++;
				if(loc == 3){ loc--; change = 1;} //if we saw SPAM, now do HAM
				else{ loc++; change = 2;} //else do SPAM
				Label next = change==1 ? Label.HAM : Label.SPAM;
				prob = p_w_given_l(i.words[s], next);
				tmp.set(loc, prob);
				wordMap.put(i.words[s], tmp);
			}
		}
	}

	/**
	 * Returns the prior probability of the label parameter, i.e. P(SPAM) or P(HAM)
	 */
	@Override
	public double p_c(Label label) {
		//prob is just #occurances/#total
		return (label==Label.SPAM ? numSpam : numHam) / (numHam+numSpam);
	}

	/**
	 * Returns the smoothed conditional probability of the word given the label,
	 * i.e. P(word|SPAM) or P(word|HAM)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {
		int classNum = label==Label.SPAM ? 1 : 0; //figure out which label to use
		double l = label==Label.SPAM ? spamTokens : hamTokens;
		double numOfThisLabel;
		//if the word is in the map, use that data
		if(wordMap.containsKey(word)){
			numOfThisLabel = wordMap.get(word).get(classNum);
		}
		//otherwise use 0
		else{
			numOfThisLabel = 0;
		}
		double delta = .00001; //smoothing size
		return ((numOfThisLabel+delta)/((vocSize*delta) + l));
	}

	/**
	 * Classifies an array of words as either SPAM or HAM. 
	 */
	@Override
	public ClassifyResult classify(String[] words) {
		ClassifyResult result = new ClassifyResult();
		Label finalClass;
		//first find prob that it's ham
		double pHam = Math.log(p_c(Label.HAM));
		double condSum = 0;
		for(int i = 0; i < words.length; i++){
			condSum += Math.log(p_w_given_l(words[i], Label.HAM));
		}
		double firstSum = pHam + condSum;
		//now find the prob that it's spam
		double pSpam = Math.log(p_c(Label.SPAM));
		double condSum1 = 0;
		for(int i = 0; i < words.length; i++){
			condSum1 += Math.log(p_w_given_l(words[i], Label.SPAM));
		}
		double secondSum = pSpam + condSum1;
		//check for the bigger one;
		if(firstSum > secondSum){
			finalClass = Label.HAM;
		}
		else{
			finalClass = Label.SPAM;
		}
		//create the classifier
		result.label = finalClass;
		result.log_prob_ham = firstSum;
		result.log_prob_spam = secondSum;
		return result;
	}
}
