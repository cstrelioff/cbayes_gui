<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1" />
<title>${title}</title>
<link rel="stylesheet" type="text/css" href="/css/main.css" />
</head>

<body>

<!-- Begin Wrapper -->
<div id="wrapper">
   
<!-- Begin Header -->
<div id="header">

${header}

</div>
<!-- End Header -->
		 
<!-- Begin Naviagtion -->
<div id="navigation">

<div class="nav">
    <ul>
        <li><a href="/">index</a></li>
        <li><a href="javascript:history.back()">Back</a></li>
%for desc in sorted(navigation.keys()):
        <li><a href="${navigation[desc]}">${desc}</a></li>
%endfor
        <li><a id="shutdown" title="Server will shutdown -- close window" href="/shutdown">Shutdown</a></li>
    </ul>
</div>

</div>
<!-- End Naviagtion -->

<!-- Begin Content -->
<div id="content">
      
${content}
		 
</div>
<!-- End Content -->
		 
</div>
<!-- End Wrapper -->
   
</body>
</html>
