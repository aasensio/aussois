; Compute the magnetic flux density using the weak-field approximation
pro computeBpar
	restore, 'stI_rebinned.idl'
	
	restore, 'stV_rebinned.idl'
	
	restore, 'wavelength_axis.sav' 
	lambda = landas[0:50]
	
	Bpar = fltarr(128,128)
	
	for i = 0, 127 do begin
		for j = 0, 127 do begin	
			st_I = stI[*,i,j]
			cont = st_I[0]
			st_I = st_I / cont
			st_V = stV[*,i,j]
			st_V = st_V / cont
			didl = 6301.d0^2 * 3.d0 * deriv(lambda, st_I)
			C = 4.6686d-13
			Bpar[i,j] = -total(st_V * didl) / (C * total(didl^2))
		endfor
	endfor
	
	save, Bpar, filename='Bpar.idl'
	
end