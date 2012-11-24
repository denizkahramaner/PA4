package cs224n.coref;

import java.util.*;

import cs224n.coref.Sentence.Token;
import cs224n.util.Pair;
import edu.stanford.nlp.util.EditDistance;

public class TokenMatchRule implements Rule{
	

	
	public Map<Mention,HashSet<Mention>> getCoreferences(Document doc){


		Map<Mention,HashSet<Mention>> clusters = new HashMap<Mention, HashSet<Mention>>();
		Map<String,Entity> names = new HashMap<String, Entity>();
		
	    for(Mention m : doc.getMentions()){
	    	for(Mention n: doc.getMentions()){
	    		if( !m.equals(n)){
	    			//Token head = m.headToken();
	    			int  count=0;
	    			
					String[] tokens1 = m.gloss().split(" ");
					String[] tokens2 = n.gloss().split(" ");
					double relevantLen =0;
//					for (String token1 : tokens1) {
//						for (String token2 : tokens2) {
//							if(token1.length() > 2){
//								relevantLen ++;
//								if ( token1.equalsIgnoreCase(token2)) {
//									count ++;
//									break;
//								}
//							}
//						}
//					}
//
//					
					
					if(tokens1.length < 4 || tokens2.length < 4){
						continue;
					}
					int start =0;
					boolean namecheck = false;
					for(int i =0; i< tokens1.length ; i++){
						for(int j =start ; j < tokens2.length; j++){
							if(tokens1[i].length() > 2){
								//relevantLen ++;
								if ( tokens1[i].equalsIgnoreCase(tokens2[j])) {
									if(Name.isName(tokens1[i])){
										//namecheck = true;
									}
									count ++;
									start =j;
								}
							}

						}
					}

					//if(relevantLen==0){
						//continue;
					//}
					relevantLen = (tokens1.length+ tokens2.length)/2;
					if((count/relevantLen) > 0.5 || namecheck){
	    			
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