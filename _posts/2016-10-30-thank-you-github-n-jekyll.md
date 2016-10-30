---
layout: post
title:  "Thank you, GitHub and Jekyll"
date:   2016-10-30 14:50:00 +0800
categories: blog
---

Thank you, GitHub and Jekyll.

Now I have my homepage again.

I'd like to share my *thoughts* and *ideas* and *source codes* here. Hope they helped.

btw, I am so happy to see Jekyll's way of managing a website. It reminds me of the old days.

Here is my naive thoughts back in 2007 (Oh my god, it was almost 10 years ago!):

> I found a very interesting method of templates when I was writing a website yesterday. 
> 
> Normal template technology is using PHP programs to read a template file, and parsing the symbols itself, than get the final > version of PHP page codes, store those to a work cache folder, and show that if need. 
> 
> The popular technology of template in PHP is like Smarty or sth. 
> 
> I got another way to using template. Anyway, it may be not so securable, but it's very easy to use. 
> 
> 1, Make HTML pages, and give them PHP extension name. 
> Like: template.php 
>
> ```php
> <html> 
> <body> 
> hello, world. 
> </body> 
> </html> 
> ```
> 
> 2, Create different pages which really need to show. 
> Like: page.php 
>
> ```php
> <? 
>   $str = hello, world. 
>   include('template.php'); 
> ?> 
> ```
> 
> 3. Replace the string using the PHP vars. 
> Like: template.php
>
> ```php
> <html> 
> <body> 
> <?=$str?> 
> </body> 
> </html> 
> ```
> 
> ok, call page.php, you can see your work. 
> 
> Addtional, you can even give the template.php a List to show the records. That's a great thing.
