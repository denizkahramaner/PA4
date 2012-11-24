package cs224n.corefsystems;

import cs224n.coref.*;
import cs224n.ling.Tree;
import cs224n.util.Pair;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.util.logging.RedwoodConfiguration;
import edu.stanford.nlp.util.logging.StanfordRedwoodConfiguration;

import java.text.DecimalFormat;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * @author Gabor Angeli (angeli at cs.stanford)
 */
@SuppressWarnings("deprecation")
public class ClassifierBased implements CoreferenceSystem {

	private Map<Document, Map<Mention, List<Mention>>> hobbsResults;
	
	private static <E> Set<E> mkSet(E[] array){
		Set<E> rtn = new HashSet<E>();
		Collections.addAll(rtn, array);
		return rtn;
	}

	private static final Set<Object> ACTIVE_FEATURES = mkSet(new Object[]{

			/*
			 * TODO: Create a set of active features
			 */

			Feature.ExactHeadMatch.class,
			//Feature.ExactMatch.class,
			//Feature.NearbyMentions.class,
			//Feature.HobbsCandidate.class,
			//Feature.NearbyEditDistance.class,
			//Feature.NearbyHeadEditDistance.class,
			Feature.PronounMatch.class,
			//Feature.GenderExactMatch.class,
			//Feature.GenderNoMatch.class,
			//Feature.AnyTokenMatch.class,

			//skeleton for how to create a pair feature
			//Pair.make(Feature.IsFeature1.class, Feature.IsFeature2.class),
	});


	private LinearClassifier<Boolean,Feature> classifier;

	public ClassifierBased(){
		StanfordRedwoodConfiguration.setup();
		RedwoodConfiguration.current().collapseApproximate().apply();
	}

	public FeatureExtractor<Pair<Mention,ClusteredMention>,Feature,Boolean> extractor = new FeatureExtractor<Pair<Mention, ClusteredMention>, Feature, Boolean>() {
		private <E> Feature feature(Class<E> clazz, Pair<Mention,ClusteredMention> input, Option<Double> count){
			
			//--Variables
			Mention onPrix = input.getFirst(); //the first mention (referred to as m_i in the handout)
			Mention candidate = input.getSecond().mention; //the second mention (referred to as m_j in the handout)
			Entity candidateCluster = input.getSecond().entity; //the cluster containing the second mention


			//--Features
			if(clazz.equals(Feature.ExactMatch.class)){
				//(exact string match)
				return new Feature.ExactMatch(onPrix.gloss().equals(candidate.gloss()));
			} else if(clazz.equals(Feature.ExactHeadMatch.class)) {
				return new Feature.ExactHeadMatch(onPrix.headWord().toLowerCase().equals(candidate.headWord().toLowerCase()));
			} else if (clazz.equals(Feature.NearbyMentions.class)) {
				/*int a = Math.abs(onPrix.beginIndexInclusive - candidate.endIndexExclusive);
				int b = Math.abs(candidate.beginIndexInclusive - onPrix.endIndexExclusive);
				int dist = Math.min(a, b); // the # of words between the two mentions*/
				int dist = Math.abs(onPrix.headWordIndex - candidate.headWordIndex);
				int NEARBY_DIST = 5;
				boolean nearby = (dist <= NEARBY_DIST);
				return new Feature.NearbyMentions(nearby);
			} else if (clazz.equals(Feature.HobbsCandidate.class)) {
				// retrieve hobbs values computed in train()
				Map<Mention, List<Mention>> candidateMentions = hobbsResults.get(onPrix.doc);
				// If it wasn't there, do Hobbs now, but store the results for the future.
				if (candidateMentions == null) {
					Map<Mention, List<Mention>> c = hobbsAlgorithm(onPrix.doc);
					hobbsResults.put(onPrix.doc, c);
					candidateMentions = c;
				}

				Map<Mention, List<Mention>> candidateMentions2 = hobbsResults.get(candidate.doc);
				// If it wasn't there, do Hobbs now, but store the results for the future.
				if (candidateMentions2 == null) {
					Map<Mention, List<Mention>> c = hobbsAlgorithm(candidate.doc);
					hobbsResults.put(candidate.doc, c);
					candidateMentions2 = c;
				}
				
				// Search through each relevant list to see if one is a
				// coreferent candidate for the other.
				boolean found = false;
				if (candidateMentions.get(onPrix) != null) {
					//System.out.println("success");
					for (Mention m : candidateMentions.get(onPrix)) {
						if (m.equals(candidate)) {
							found = true;
							break;
						}
					}
				}
				if (!found && candidateMentions2.get(candidate) != null) {
					//System.out.println("success");
					for (Mention m : candidateMentions2.get(candidate)) {
						if (m.equals(onPrix)) {
							found = true;
							break;
						}
					}
				}
				return new Feature.HobbsCandidate(found);
			} else if (clazz.equals(Feature.NearbyEditDistance.class)) {
				int d = editDistance(onPrix.gloss().toLowerCase(), candidate.gloss().toLowerCase());
				int NEARBY_EDIT_DIST = 3; // 1 is good, and 2 is terribad :(
				return new Feature.NearbyEditDistance(d <= NEARBY_EDIT_DIST);
			} else if (clazz.equals(Feature.NearbyHeadEditDistance.class)) {
				int d = editDistance(onPrix.headWord().toLowerCase(), candidate.headWord().toLowerCase());
				int NEARBY_EDIT_DIST = 3;
				return new Feature.NearbyHeadEditDistance(d <= NEARBY_EDIT_DIST);
			} else if (clazz.equals(Feature.PronounMatch.class)) {
				boolean match = false;
				if (Pronoun.isSomePronoun(onPrix.headWord()) && Pronoun.isSomePronoun(candidate.headWord())) {
					// both are pronouns
					// do they match speaker and gender and number? (ignore type)
					Pronoun p = Pronoun.valueOrNull(onPrix.headWord());
					Pronoun p2 = Pronoun.valueOrNull(candidate.headWord());
					if (p != null && p2 != null) // apparently the p and p2 can still be null since the set and enum aren't the same
						match = (p.gender == p2.gender && p.speaker == p2.speaker && p.plural == p2.plural );
				}
				return new Feature.PronounMatch(match);
			} else if (clazz.equals(Feature.GenderExactMatch.class)) {
				boolean match = false;
				
				// require matching gender
				Gender g1 = Gender.NEUTRAL;
				Gender g2 = Gender.EITHER;
				if (Pronoun.isSomePronoun(onPrix.headWord())) {
					Pronoun p = Pronoun.valueOrNull(onPrix.headWord());
					if (p != null)
						g1 = p.gender;
				}
				if (Pronoun.isSomePronoun(candidate.headWord())) {
					Pronoun p = Pronoun.valueOrNull(candidate.headWord());
					if (p != null)
						g2 = p.gender;
				}
				if (Name.isName(onPrix.headWord())) {
					g1 = Name.gender(onPrix.headWord());
				}
				if (Name.isName(candidate.headWord())) {
					g2 = Name.gender(candidate.headWord());
				}
				match = (g1 == g2);
				
				return new Feature.GenderExactMatch(match);
			} else if (clazz.equals(Feature.GenderNoMatch.class)) {
				boolean match = false;
				
				// require matching gender
				Gender g1 = Gender.NEUTRAL;
				Gender g2 = Gender.EITHER;
				if (Pronoun.isSomePronoun(onPrix.headWord())) {
					Pronoun p = Pronoun.valueOrNull(onPrix.headWord());
					if (p != null)
						g1 = p.gender;
				}
				if (Pronoun.isSomePronoun(candidate.headWord())) {
					Pronoun p = Pronoun.valueOrNull(candidate.headWord());
					if (p != null)
						g2 = p.gender;
				}
				if (Name.isName(onPrix.headWord())) {
					g1 = Name.gender(onPrix.headWord());
				}
				if (Name.isName(candidate.headWord())) {
					g2 = Name.gender(candidate.headWord());
				}
				// require not compatible genders
				match = (!g1.isCompatible(g2));
				
				return new Feature.GenderNoMatch(match);
			} else if(clazz.equals(Feature.AnyTokenMatch.class)) {
				boolean match = false;
				
				String[] tokens1 = onPrix.gloss().split(" ");
				String[] tokens2 = candidate.gloss().split(" ");
				for (String token1 : tokens1) {
					for (String token2 : tokens2) {
						if (token1.length() > 1 && token1.equalsIgnoreCase(token2)) {
							match = true;
							break;
						}
					}
				}
				
				return new Feature.AnyTokenMatch(match);
//			} else if(clazz.equals(Feature.NewFeature.class) {
				/*
				 * TODO: Add features to return for specific classes. Implement calculating values of features here.
				 */
			}
			else {
				throw new IllegalArgumentException("Unregistered feature: " + clazz);
			}
		}

		@SuppressWarnings({"unchecked"})
		@Override
		protected void fillFeatures(Pair<Mention, ClusteredMention> input, Counter<Feature> inFeatures, Boolean output, Counter<Feature> outFeatures) {
			//--Input Features
			for(Object o : ACTIVE_FEATURES){
				if(o instanceof Class){
					//(case: singleton feature)
					Option<Double> count = new Option<Double>(1.0);
					Feature feat = feature((Class) o, input, count);
					if(count.get() > 0.0){
						inFeatures.incrementCount(feat, count.get());
					}
				} else if(o instanceof Pair){
					//(case: pair of features)
					Pair<Class,Class> pair = (Pair<Class,Class>) o;
					Option<Double> countA = new Option<Double>(1.0);
					Option<Double> countB = new Option<Double>(1.0);
					Feature featA = feature(pair.getFirst(), input, countA);
					Feature featB = feature(pair.getSecond(), input, countB);
					if(countA.get() * countB.get() > 0.0){
						inFeatures.incrementCount(new Feature.PairFeature(featA, featB), countA.get() * countB.get());
					}
				}
			}

			//--Output Features
			if(output != null){
				outFeatures.incrementCount(new Feature.CoreferentIndicator(output), 1.0);
			}
		}

		@Override
		protected Feature concat(Feature a, Feature b) {
			return new Feature.PairFeature(a,b);
		}
	};

	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		// NEW:
		hobbsResults = new HashMap<Document, Map<Mention, List<Mention>>>();
		
		
		startTrack("Training");
		//--Variables
		RVFDataset<Boolean, Feature> dataset = new RVFDataset<Boolean, Feature>();
		LinearClassifierFactory<Boolean, Feature> fact = new LinearClassifierFactory<Boolean,Feature>();
		//--Feature Extraction
		startTrack("Feature Extraction");
		for(Pair<Document,List<Entity>> datum : trainingData){
			//(document variables)
			Document doc = datum.getFirst();
			// NEW:
			hobbsResults.put(doc, hobbsAlgorithm(doc));
			
			List<Entity> goldClusters = datum.getSecond();
			List<Mention> mentions = doc.getMentions();
			Map<Mention,Entity> goldEntities = Entity.mentionToEntityMap(goldClusters);
			startTrack("Document " + doc.id);
			//(for each mention...)
			for(int i=0; i<mentions.size(); i++){
				//(get the mention and its cluster)
				Mention onPrix = mentions.get(i);
				Entity source = goldEntities.get(onPrix);
				if(source == null){ throw new IllegalArgumentException("Mention has no gold entity: " + onPrix); }
				//(for each previous mention...)
				int oldSize = dataset.size();
				for(int j=i-1; j>=0; j--){
					//(get previous mention and its cluster)
					Mention cand = mentions.get(j);
					Entity target = goldEntities.get(cand);
					if(target == null){ throw new IllegalArgumentException("Mention has no gold entity: " + cand); }
					//(extract features)
					Counter<Feature> feats = extractor.extractFeatures(Pair.make(onPrix, cand.markCoreferent(target)));
					//(add datum)
					dataset.add(new RVFDatum<Boolean, Feature>(feats, target == source));
					//(stop if
					if(target == source){ break; }
				}
				//logf("Mention %s (%d datums)", onPrix.toString(), dataset.size() - oldSize);
			}
			endTrack("Document " + doc.id);
		}
		endTrack("Feature Extraction");
		//--Train Classifier
		startTrack("Minimizer");
		this.classifier = fact.trainClassifier(dataset);
		endTrack("Minimizer");
		//--Dump Weights
		startTrack("Features");
		//(get labels to print)
		Set<Boolean> labels = new HashSet<Boolean>();
		labels.add(true);
		//(print features)
		for(Triple<Feature,Boolean,Double> featureInfo : this.classifier.getTopFeatures(labels, 0.0, true, 100, true)){
			Feature feature = featureInfo.first();
			Boolean label = featureInfo.second();
			Double magnitude = featureInfo.third();
			log(FORCE,new DecimalFormat("0.000").format(magnitude) + " [" + label + "] " + feature);
		}
		end_Track("Features");
		endTrack("Training");
	}

	public List<ClusteredMention> runCoreference(Document doc) {
		//--Overhead
		startTrack("Testing " + doc.id);
		//(variables)
		List<ClusteredMention> rtn = new ArrayList<ClusteredMention>(doc.getMentions().size());
		List<Mention> mentions = doc.getMentions();
		int singletons = 0;
		//--Run Classifier
		for(int i=0; i<mentions.size(); i++){
			//(variables)
			Mention onPrix = mentions.get(i);
			int coreferentWith = -1;
			//(get mention it is coreferent with)
			for(int j=i-1; j>=0; j--){
				ClusteredMention cand = rtn.get(j);
				boolean coreferent = classifier.classOf(new RVFDatum<Boolean, Feature>(extractor.extractFeatures(Pair.make(onPrix, cand))));
				if(coreferent){
					coreferentWith = j;
					break;
				}
			}
			//(mark coreference)
			if(coreferentWith < 0){
				singletons += 1;
				rtn.add(onPrix.markSingleton());
			} else {
				//log("Mention " + onPrix + " coreferent with " + mentions.get(coreferentWith));
				rtn.add(onPrix.markCoreferent(rtn.get(coreferentWith)));
			}
		}
		//log("" + singletons + " singletons");
		//--Return
		endTrack("Testing " + doc.id);
		return rtn;
	}

	private class Option<T> {
		private T obj;
		public Option(T obj){ this.obj = obj; }
		public Option(){};
		public T get(){ return obj; }
		public void set(T obj){ this.obj = obj; }
		public boolean exists(){ return obj != null; }
	}
	
	
	private Map<Tree<String>, Tree<String>> getChildParentMap(Tree<String> tree) {
		return getChildParentMapHelper(tree, new HashMap<Tree<String>, Tree<String>>());
	}
	private Map<Tree<String>, Tree<String>> getChildParentMapHelper(Tree<String> tree, Map<Tree<String>, Tree<String>> childParentMap) {
		if (tree == null)
			return childParentMap;
		
		for (Tree<String> child : tree.getChildren()) {
			childParentMap.put(child, tree);
			childParentMap = getChildParentMapHelper(child, childParentMap);
			// I don't think assigning is necessary since the same map is being mutated the whole time, but w/e
		}
		
		return childParentMap;
	}
	private boolean hasSAmongParents(Tree<String> xTree, Map<Tree<String>, Tree<String>> childToParent) {
		// returns false when
		// xTree was the highest S in the sentence. Well, more like there were no more S's in the sentence
		boolean found = false;
		Tree<String> temp = xTree;
		while(childToParent.get(temp) != null) {
			temp = childToParent.get(temp);
			if (temp.getLabel() == "S") {
				found = true;
				break;
			}
		}
		
		return found;
	}
	
	private List<Tree<String>> findNPAboveNPorS(Tree<String> head) {
		return findNPAboveNPorSHelper(head, new ArrayList<Tree<String>>());
	}
	private List<Tree<String>> findNPAboveNPorSHelper(Tree<String> head, List<Tree<String>> list) {
		if (head == null)
			return list;
		
		// check head
		if (head.getLabel().equals("NP")) {
			list.add(head);
			return list;
		} else if (head.getLabel().equals("S")) {
			return list;
		}
		
		// check children
		for (Tree<String> child : head.getChildren()) {
			list = findNPAboveNPorSHelper(child, list);
		}
		
		return list;
	}
	
	private Map<Mention, List<Mention>> hobbsAlgorithm(Document doc) {
		// Let's try to do Hobb's. Every mention needs to take a look at who it could be paired with.
	    Map<Mention, List<Mention>> candidateMentions = new HashMap<Mention,List<Mention>>();
	    
	    Map<Tree<String>, Mention> treeMentionMap = new HashMap<Tree<String>, Mention>();
	    for (Mention m : doc.getMentions()) {
	    	treeMentionMap.put(m.parse, m);
	    }
	    
		for(Mention m : doc.getMentions()) {
			List<Mention> hobbs = new ArrayList<Mention>();
			
			Tree<String> sentenceTree = m.sentence.parse;
			
			Map<Tree<String>, Tree<String>> childToParent = getChildParentMap(sentenceTree);
			
			// 1 Begin at NP of our mention
			Tree<String> mentionTree = m.parse;
			
			// 2 Go up the tree to the first S or NP, this is called the xTree
			//   The path used pPath will just be the node whose parent is xTree
			Tree<String> xTree = childToParent.get(mentionTree);
			Tree<String> pPath = mentionTree;
			while (xTree != null && !xTree.getLabel().equals("NP") && !xTree.getLabel().equals("S")) {
				pPath = xTree;
				xTree = childToParent.get(xTree);
			}
			if (xTree == null) {
				//System.out.println("2: hit top of sentence w/o finding another NP or S");
				continue; // Well there's nothing we can do we hit the top of the sentence already...
			}
			
			// 3 Search the tree left of pPath breadth first
			for (Tree<String> child : xTree.getChildren()) {
				// We have gone as far right as possible if...
				if (child == pPath)
					break;
				
				// Search to your heart's content for any NP. These work if you can find a parent that isn't X that is NP or S
				for (Tree<String> node : child.getPreOrderTraversal()) {
					if (node.getLabel().equals("NP")) {
						// Great. Now does it have a parent that is either NP or S (before hitting X?)
						
						boolean found = false;
						Tree<String> parent = node;
						while (!parent.equals(xTree) && !found) {
							parent = childToParent.get(parent);
							
							if (parent == null) // NOTE: this probably should not be needed...
								break;
							
							if (parent.getLabel().equals("NP") || parent.getLabel().equals("S")) {
								found = true;
							}
						}
						
						if (found) {
							// we found a candidate! Get the mention for the tree and add it!
							Mention m2 = treeMentionMap.get(node);
							if (m2 != null) { hobbs.add(m2); }
						}
					}
				}
			}
			
			// 4 decide if you need to enter this loop
			while (hasSAmongParents(xTree, childToParent)){
				// 5-9 while loop
				
				// 5 Go up the tree to the first S or NP, this is called the xTree
				//   The path used pPath will just be the node whose parent is xTree
				xTree = childToParent.get(xTree);
				pPath = mentionTree;
				while (xTree != null && !xTree.getLabel().equals("NP") && !xTree.getLabel().equals("S")) {
					pPath = xTree;
					xTree = childToParent.get(xTree);
				}
				if (xTree == null) {
					System.out.println("5: hit top of sentence w/o finding another NP or S. BAD");
					continue; // Well there's nothing we can do we hit the top of the sentence already...
					// NOTE THAT THIS ONE SHOULDN"T HAPPEN
				}
				
				// 6 If X and the path passes through the nominal that dominates the mention, don't mark it. Else, mark it.
				//   It seems like it'll never do this, so let's mark it anyway for now
				if (xTree.getLabel().equals("NP")) { // requires NP
					Mention m2 = treeMentionMap.get(xTree);
					if (m2 != null) { hobbs.add(m2); }
				}
				
				// 7 Traverse branches to the left of p, breadth first. Any NP counts as an antecedent
				for (Tree<String> child : xTree.getChildren()) {
					// We have gone as far right as possible if...
					if (child == pPath)
						break;
					
					// Search to your heart's content for any NP. These work if you can find a parent that isn't X that is NP or S
					for (Tree<String> node : child.getPreOrderTraversal()) {
						if (node.getLabel().equals("NP")) {
							// we found a candidate! Get the mention for the tree and add it!
							Mention m2 = treeMentionMap.get(node);
							if (m2 != null) { hobbs.add(m2); }
						}
					}
				}
				
				// 8 If X is an S node, traverse branches of X to the right of P. But stop upon finding any NP or S
				if (xTree.getLabel().equals("S")) {
					boolean seenP = false;
					for (Tree<String> child : xTree.getChildren()) {
						if (!seenP) {
							if (child != pPath)
								continue;
							else
								seenP = true;
						}
						
						// get the list of NP's that happened to not be below any NP or S
						List<Tree<String>> candidates = findNPAboveNPorS(child);
						for (Tree<String> candidate : candidates) {
							// get the corresponding mention and add it to the list
							Mention m2 = treeMentionMap.get(candidate);
							if (m2 != null) { hobbs.add(m2); }
						}
						
					}
				}
				
				// 9 go back to 4
			}
			// Completion of 4: check previous sentence(s) (breadth first) for the first NP to propose as antecedent
			for (int i = 0; i < doc.sentences.size(); i++) {
				if (doc.sentences.get(i) == m.sentence && i > 0) {
					// then we have a previous sentence to explore
					
					Tree<String> prevTree = doc.sentences.get(i - 1).parse;
					for (Tree<String> child : prevTree.getPreOrderTraversal()) {
						if (child.getLabel().equals("NP")) {
							// add this to the list and stop!
							Mention m2 = treeMentionMap.get(child);
							if (m2 != null) { hobbs.add(m2); }
							break;
						}
					}
					break;
				}
			}
			
			candidateMentions.put(m, hobbs);
		}
	    
	    return candidateMentions;
	}

	// The following code was free from:
	// http://professorjava.weebly.com/edit-distance.html
	public static int editDistance(String s, String t){
	    int m=s.length();
	    int n=t.length();
	    int[][]d=new int[m+1][n+1];
	    for(int i=0;i<=m;i++){
	      d[i][0]=i;
	    }
	    for(int j=0;j<=n;j++){
	      d[0][j]=j;
	    }
	    for(int j=1;j<=n;j++){
	      for(int i=1;i<=m;i++){
	        if(s.charAt(i-1)==t.charAt(j-1)){
	          d[i][j]=d[i-1][j-1];
	        }
	        else{
	          d[i][j]=min((d[i-1][j]+1),(d[i][j-1]+1),(d[i-1][j-1]+1));
	        }
	      }
	    }
	    return(d[m][n]);
	  }
	  public static int min(int a,int b,int c){
	    return(Math.min(Math.min(a,b),c));
	  }
	
}
