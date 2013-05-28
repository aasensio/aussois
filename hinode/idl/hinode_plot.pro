;-----------------------------------------
; Show images
;-----------------------------------------
pro show_images, state	
	widget_control, state.plotleftWidget, GET_VALUE=leftWin	 			
	wset, leftWin
	
	if (state.image eq 0) then begin
		loadct,0,/silent
		tvscl, rebin(reform((*state.stI)[0,*,*]),512,512)
	endif else begin
		loadct,0,/silent
		tvscl, reform((*state.Bpar))
	endelse
end


;-----------------------------------------
; Plot profiles
;-----------------------------------------
pro plot_profiles, state, x, y	
	widget_control, state.plotrightWidget, GET_VALUE=rightWin
	wset, rightWin
	
	!p.multi = [0,1,2]
	stI = (*state.stI)[*,x/4,y/4]
	cont = stI[0]
	stI = stI / cont
	stV = (*state.stV)[*,x/4,y/4]
	stV = stV / cont
	Bpar = (*state.Bpar)[x/4,y/4]
	
	C = 4.6686d-13
	dIdl = 6301.d0^2 * 3.d0 * deriv((*state.lambda), stI)
		
	plot, (*state.lambda), stI, ytit='I/I!dc', tit='X: '+strtrim(string(x),2)+' - Y: '+strtrim(string(y),2), xsty=1, xtit='Wavelength [A]'
		
	cwpal
	plot, (*state.lambda), 100.d0*stV, ytit='V/I!dc [%]', tit='Mag. flux density = '+strtrim(string(Bpar),2), xsty=1, xtit='Wavelength [A]'
	oplot, (*state.lambda), -100.d0 * C * Bpar * didl, col=2
	loadct,0,/silent	
	
	!p.multi = 0
end