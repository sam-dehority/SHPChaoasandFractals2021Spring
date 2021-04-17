def plot_iterates(f,num_iterates, x, color = 'blue'):
    pts = point((x.real(), x.imag()), color = color)
    for i in range(1, num_iterates):
        new = iterate(f, i, x)
        if abs(new) < 10:
            pts = pts + point((new.real(), new.imag()), color = color)
    return(pts)


import numpy as np
import numpy.polynomial.polynomial as poly
def plot_roots_N(f, color = 'blue'):
    j=np.complex(0,1)
    cs = np.array(R(f).coefficients(sparse=False)) # gets the coefficients of the polynomial f
    roots_candidates = [CDF(a) for a in list(poly.polyroots(cs))]# poly.polyroots finds the roots numerically. CDF makes the sage CDFs
    roots = [a for a in roots_candidates if abs(f(z = a)) < 0.1]
    points = [point((a.real(), a.imag()), color = color) for a in roots ]
    return sum(points)

from PIL import Image as pilimg

def plot_with_julia(graphics, f, **kwds):
    """
    This function returns a plot of the graphics overlayed with the julia set of the function f
    
    -- graphics: a sage graphics object, e.g. points
    
    -- f : A polynomial, e.g. z^2 + 0.25
    
    -- **kwds: the same keywords as for julia_plot()
    
    EXAMPLE ::
    
    sage: a = point((0,0), color='red')
    sage: R.<z> = CDF[]
    sage: f = z^2 + 0.25
    sage: plot_with_julia(a, f)
    
    """
    kwdscopy = dict(kwds)
    x_center = kwds.pop("x_center", 0.0)
    y_center = kwds.pop("y_center", 0.0)
    image_width = kwds.pop("image_width", 4.0)
    pixel_count = kwds.pop("pixel_count", 500)
    xmax = x_center + image_width*0.5
    xmin = x_center - image_width*0.5
    ymax = y_center + image_width*0.5
    ymin = y_center - image_width*0.5
    graphics.save("./tmp/g.png", transparent = True, axes=False, xmin=xmin, xmax = xmax, ymin = ymin, ymax = ymax, aspect_ratio = 1, figsize = [80,80])
    julia = julia_plot(f, mandelbrot = False, **kwdscopy)
    julia_modified = julia.pil.convert('RGBA')
    mod = pilimg.open('./tmp/g.png')
    mod_resized = mod.resize((julia.height(), julia.width()))
    julia_modified.alpha_composite(mod_resized)
    return julia_modified

def plot_with_mandelbrot(graphics, **kwds):
    """
    This function returns a plot of the graphics overlayed with the mandelbrot set
    
    -- graphics: a sage graphics object, e.g. points

    -- **kwds: the same keywords as for julia_plot()
    
    EXAMPLE ::
    
    sage: a = point((0,0.25), color='white')
    sage: plot_with_mandelbrot(a, image_width = 3.0, pixel_count = 1000)
    
    """
    kwdscopy = dict(kwds)
    x_center = kwds.pop("x_center", 0.0)
    y_center = kwds.pop("y_center", 0.0)
    image_width = kwds.pop("image_width", 4.0)
    pixel_count = kwds.pop("pixel_count", 500)
    xmax = x_center + image_width*0.5
    xmin = x_center - image_width*0.5
    ymax = y_center + image_width*0.5
    ymin = y_center - image_width*0.5
    graphics.save("./tmp/g.png", transparent = True, axes=False, xmin=xmin, xmax = xmax, ymin = ymin, ymax = ymax, aspect_ratio = 1, figsize = [10,10])
    m = mandelbrot_plot(**kwdscopy)
    m_modified = m.pil.convert('RGBA')
    mod = pilimg.open('./tmp/g.png')
    mod_resized = mod.resize((m.height(), m.width()))
    m_modified.alpha_composite(mod_resized)
    return m_modified

def mapped_circle(center, radius, f, num_circles = 5, num_splines = 10, color = 'orange', thickness = .4, fillalpha= .1):
    """
    Returns the image of a circle with radial coordinate lines under a given map f. 
    INPUT:

    - ``center`` -- complex numer

    - ``radius`` -- real number
    
    - ``f`` -- a function of a variable `z` 
    
    OUTPUT:
    
    -  A graphics object of the image of the image f(C) of the circle C with given radius and center. 

    
    """
    center_coords = (CDF(center).real(), CDF(center).imag())
    radii = [radius/num_circles * k for k in range(1, num_circles + 1)]
    angles = [2*pi *θ / (num_splines) for θ in range(2*num_splines)]
    coords_f = lambda x,y : ((q := f(z = x + I*y)).real(), q.imag())
    t = var('t')
    spline = lambda θ : parametric_plot(coords_f(x = cos(θ)*t + center_coords[0] ,y =  sin(θ)*t + center_coords[1]), (t , 0, radius), color = color, thickness = thickness)
    circ = lambda r : parametric_plot(coords_f(x = r*cos(t) + center_coords[0], y = r*sin(t) + center_coords[1]), (t,0,2*pi), color = color, fill = True, fillcolor = color, fillalpha = fillalpha, thickness = thickness)
    return sum(spline(θ) for θ in angles) + sum(circ(r) for r in radii)

def back_iterates(L,c, tolerance = 0.01):
    """
    returns all points z such that z^2_+ c is in L
    uses that f_c^{-1}(t) = sqrt(t - c). 
    
    Also we remove points in the resulting list so that no two points are closer than tolerance away from eachother
    """
    out1 = []
    for l in L:
        lminusc = CDF(l - c)
        out1 += lminusc.sqrt(all = True)
    out2 = []
    for o in out1:
        if not any(abs(z - o) < tolerance for z in out2):
            out2.append(o)
    return out2    

def curves(L, tolerance = 0.3):
    """
    For a list L of points, (i.e. actually complex numbers) returns a list of lists
    [L1, L2, ...]
    where each list Li has the property that nearby points in Li are close to eachother, and that points in 
    Li are far away from from Lj if i != j, where close means closer than tolerance, and far away means not close
    """
    out = []
    new = [L[0]]
    pt = new[0]
    L = L[1:]
    while L != []:
        with_distances = [(l, abs(pt - l)) for l in L]
        distances = [d for (l,d) in with_distances]
        L = [l for (l,d) in  with_distances]
        if len(L) == 0:
            break
        min_distance = min(distances)
        closest_index = distances.index(min_distance)
        if min_distance > tolerance:
            out.append(new)
            new = [L[0]]
            pt = new[0]
            L = L[1:]
        else:
            pt = L[closest_index]
            new = new  + [pt]
            L = L[:closest_index] + L[closest_index+1 :]
    out.append(new)
    return out

def back_iterates_n(L, c, n , tolerance = 0.01):
    """
    Returns a list of lists [L1, ... , Ln]
    which are the first n back iterates of L under f_c
    """
    iterate_list = [L]
    for i in range(n + 1):
        iterate_list.append(back_iterates(iterate_list[-1], c, tolerance = tolerance))
    return iterate_list[1:]

def plot_backiterated_circles(c, R, num_circles, back_tolerance = 0.01, curve_tolerance = 0.2, starting_points = 20):
    """c
    Plots the backiterated circles, starting with a circle with radius R, with 20 starting points.
    """
    L = circle_points(radius = R, num_points = starting_points)
    back = back_iterates_n(L, c, num_circles, tolerance = back_tolerance)[1:]
    cs = [c for b in back for c in curves(b, tolerance = curve_tolerance)]
    level_plots = [list_plot(a,plotjoined = True) for a in cs]
    return sum(level_plots)

def external_ray_points_backiterated(c, qs, R_start = 15, iterates = 30):
    rays = [[R_start*exp(I*2*pi*q)] for q in qs]
    step = [r[-1] for r in rays]
    for i in range(iterates):
        back_iterates_array = [back_iterates([a], c) for a in step]
        step2 = []
        for t in range(len(step)):
            candidates = back_iterates_array[(t + 1 if t != len(step)-1 else 0)]
            (c0, c1) = candidates[0], candidates[1]
            ξ = step[t]
            if abs(ξ - c0) < abs(ξ - c1):
                step2.append(c0)
                rays[t].append(c0)
            else:
                step2.append(c1)
                rays[t].append(c1)
        step = step2
    return [r[1:] for r in rays]

def external_ray_points(c, qs, R_start = 15, iterates = 40): # So we want to actually include some intermediate points, by starting and various radii
    starting_radii = [R_start^(1 - 0.05*j) for j in range(7)]
    poorly_ordered_points = [external_ray_points_backiterated(c,qs, R_start = r, iterates = iterates) for r in starting_radii]
    rays = []
    for i in range(len(qs)):
        rayi = []
        for l in range(iterates):
            for j in range(7):
                rayi.append(poorly_ordered_points[j][i][l])
        rays.append(rayi)
    return rays