package cs224n.coref;

import java.util.*;
public class DistanceFilter implements Filter{
	
	public Map<Mention,HashSet<Mention>> filter(Map<Mention,HashSet<Mention>> mentions,Document doc){
		
		for(Mention m : mentions.keySet()){
			Iterator<Mention> it = mentions.get(m).iterator();
			while(it.hasNext()){
				Mention n = it.next();
				if(Math.abs(doc.indexOfSentence(m.sentence) - doc.indexOfSentence(n.sentence)) > 4 ){
					it.remove();
					continue;
				}
				if(Math.abs(doc.indexOfMention(m) - doc.indexOfMention(n)) > ((double)doc.getMentions().size()/doc.sentences.size())){
					it.remove();
					continue;
				}

				//if( m.sentence.equals(n.sentence) &&((m.endIndexExclusive <= n.beginIndexInclusive)&&(n.beginIndexInclusive-m.endIndexExclusive) > 2)|| ((n.endIndexExclusive <= m.beginIndexInclusive)&&(m.beginIndexInclusive-n.endIndexExclusive) > 2)){
					//it.remove();
				//}
				//if(Math.abs(m.headWordIndex-n.headWordIndex) >8){
					//it.remove();
					//continue;
				//}
				
			}
		}
		
		return mentions;
	}
	
}