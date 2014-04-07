<table class="hoverTable">
<%
cumm_pr = 0.    
%>

<tr>
  <td>Model Topology</td>
  <td>Probability</td>
  <td>Cumulative Probability</td>
</tr>
% for k in sorted(adict, key=adict.get, reverse=True)[:nM]:
    <tr><td>${k}</td><td>${adict[k]}</td><td>${cumm_pr}</td></tr>
% endfor
</table>
