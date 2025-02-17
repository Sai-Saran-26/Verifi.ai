import React from 'react'
import './Navbar.css'
export const Navbar = () => {
  return (
    <div className='Nav-bar'>
        <div className='logo'>
            <span>Verifi.ai</span>
        </div>
        <div className='Nav-bar-list'>
                <span>Home</span>
                <span>Contact Us</span>
                <span>How it Works ?</span>
        </div>
        <div className='Nav-btn'>
            <button>Try now</button>
        </div>
    </div>
  )
}
