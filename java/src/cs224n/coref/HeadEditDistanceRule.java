package cs224n.coref;

import java.util.*;

import cs224n.coref.Sentence.Token;
import cs224n.util.Pair;
import edu.stanford.nlp.util.EditDistance;

public class HeadEditDistanceRule implements Rule{
	
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
		
	
	public Map<Mention,HashSet<Mention>> getCoreferences(Document doc){


		Map<Mention,HashSet<Mention>> clusters = new HashMap<Mention, HashSet<Mention>>();
		Map<String,Entity> names = new HashMap<String, Entity>();
		
	    for(Mention m : doc.getMentions()){
	    	for(Mention n: doc.getMentions()){
	    		if( !m.equals(n)){
	    			//Token head = m.headToken();
	    			boolean match=false;
	    			int dist = editDistance(m.headWord(), n.headWord());
	    			double den = (m.headWord().length() + n.headWord().length())/2;
	    			
	    			if(((double)dist/den) < 0.4 && Math.min(m.headWord().length(), n.headWord().length()) > 4 && m.headToken().isNoun() && n.headToken().isNoun()){
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