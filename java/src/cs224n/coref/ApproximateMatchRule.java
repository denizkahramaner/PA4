package cs224n.coref;

import java.util.*;

public class ApproximateMatchRule implements Rule{
	
	public Map<Mention,HashSet<Mention>> getCoreferences(Document doc){
		Map<Mention,HashSet<Mention>> clusters = new HashMap<Mention, HashSet<Mention>>();
		Map<String,Entity> names = new HashMap<String, Entity>();
		
	    for(Mention m : doc.getMentions()){
	    	for(Mention n: doc.getMentions()){
	    		if(m.headWord().indexOf(n.headWord()) > 0 && !m.equals(n)){
	    			
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

	    return clusters;
		
	}
	

}