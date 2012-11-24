package cs224n.coref;

import java.util.*;

import cs224n.coref.Sentence.Token;
import cs224n.util.Pair;

public class NaivePronounRule implements Rule{
	
	public Map<Mention,HashSet<Mention>> getCoreferences(Document doc){

		Map<Mention,HashSet<Mention>> clusters = new HashMap<Mention, HashSet<Mention>>();
		Map<String,Entity> names = new HashMap<String, Entity>();
		
	    for(Mention m : doc.getMentions()){
	    	for(Mention n: doc.getMentions()){
	    		if( !m.equals(n)){
	    			//Token head = m.headToken();
	    			boolean match=false;
	    			if ((Pronoun.isSomePronoun(m.headWord()) && Pronoun.isSomePronoun(n.headWord())) || (Pronoun.isSomePronoun(m.headWord()) && n.headToken().isNoun() && doc.indexOfMention(m) > doc.indexOfMention(n)) || (Pronoun.isSomePronoun(n.headWord()) && m.headToken().isNoun() && doc.indexOfMention(n)>doc.indexOfMention(m))) {
	    				match =true;
	    			}
	    			if(match){
	    				if(clusters.containsKey(m)){
	    					clusters.get(m).add(n);
	    				}
	    			
	    				else{
	    					clusters.put(m, new HashSet<Mention>());
	    	
	    					clusters.get(m).add(n);
	    				}
	    			}
	    		}
	    	}

	      }
	    return clusters;
		
	}
	
	
}