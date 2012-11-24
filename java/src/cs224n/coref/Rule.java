package cs224n.coref;

import java.util.*;

public interface Rule {
	
	public Map<Mention,HashSet<Mention>> getCoreferences(Document d); 
		
}